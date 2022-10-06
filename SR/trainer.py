import os
import math
import utility
import torch
import numpy as np
from decimal import Decimal
import torch.nn.functional as F


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # learning rate schedule
        lr = 1e-4 * (2 ** -(epoch // 10))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (img_lr, img_hr, _, idx_scale) in enumerate(self.loader_train):
            img_lr, img_hr = self.prepare(img_lr, img_hr)

            # update tau for gumbel softmax
            tau = max(1 * (0.001 ** (((epoch-1) * 1000 + batch) / 10 / 1000)), 0.001)
            for m in self.model.modules():
                if hasattr(m, '_update_tau'):
                    m.tau = tau

            # initialization
            if epoch == 1 and batch == 0:
                self.model.model.module(img_lr[0:1, ...])

            # inference
            self.optim.zero_grad()
            img_sr = self.model(img_lr, idx_scale)

            # loss function
            loss = self.loss(img_sr, img_hr)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if (epoch) % 1 == 0:
            target = self.model
            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                for idx_img, (img_lr, img_hr, filename, _) in enumerate(self.loader_test):
                    img_lr, img_hr = self.prepare(img_lr, img_hr)
                    img_lr, img_hr = self.crop_border(img_lr, img_hr, scale)

                    # inferencce
                    img_sr = self.model(img_lr, idx_scale)

                    # quantization to int8
                    img_sr = utility.quantize(img_sr, self.args.rgb_range)
                    img_hr = utility.quantize(img_hr, self.args.rgb_range)
                    save_list = [img_sr]

                    # calculate metrics
                    eval_psnr += utility.calc_psnr(
                        img_sr, img_hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        img_sr, img_hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    # save SR results
                    if self.args.save_results:
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        eval_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    ))

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def crop_border(self, img_lr, img_hr, scale):
        N, C, H, W = img_lr.size()
        H = H // 4 * 4
        W = W // 4 * 4

        img_lr = img_lr[:, :, :H, :W]
        img_hr = img_hr[:, :, :round(scale * H), :round(scale * W)]

        return img_lr, img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

