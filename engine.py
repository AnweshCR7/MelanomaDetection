"""
__author__: Anwesh Marwade
__Inspired By__: Abhishek Thakur
"""
import torch
from avg_meter import AverageMeter
from tqdm import tqdm


class Engine:
    def __init__(
            self,
            model,
            optimizer,
            device,
            scheduler=None,
            accumulation_steps=1,
            use_tpu=False,
            tpu_print=10,
            model_fn=None,
            use_mean_loss=False,
    ):
        """
        model_fn is a custom function that takes batch of data, device and model and returns loss
        for example:
            def model_fn(data, device, model):
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                _, loss = model(images, targets)
                return loss
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        # Step jump to accumulate for the gradient computation
        self.accumulation_steps = accumulation_steps
        self.use_tpu = use_tpu
        self.tpu_print = tpu_print
        self.model_fn = model_fn
        self.use_mean_loss = use_mean_loss
        self.scaler = None
        # Print warnings in case of missing libraries

    def train(self, data_loader):
        losses = AverageMeter()

        # Set train mode
        self.model.train()

        # Reset optimizer weights
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()

        # init tqdm loop
        tq = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tq):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()

            if self.model_fn is None:
                # data.item is a dict of image and targets
                for key, value in data.items():
                    data[key] = value.to(self.device)
                _, loss = self.model(**data)

            else:
                loss = self.model_fn(data, self.device, self.model)

            with torch.set_grad_enabled(True):
                if self.use_mean_loss:
                    loss = loss.mean()

                loss.backward()

                if (b_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                if b_idx > 0:
                    self.optimizer.zero_grad()

            # Update the avg meter
            losses.update(loss.item(), data_loader.batch_size)
            tq.set_postfix(loss=losses.avg)

        tq.close()
        return losses.avg

    def evaluate(self, data_loader, return_predictions=False):
        losses = AverageMeter()

        self.model.eval()
        final_preds = []
        with torch.no_grad():
            tq = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tq):
                for key,value in data.items():
                    data[key] = value.to(self.device)

                batch_preds, loss = self.model(**data)
                if return_predictions:
                    final_preds.append(batch_preds)

                if self.use_mean_loss:
                    loss = loss.mean()

                losses.update(loss.item(), data_loader.batch_size)
                tq.set_postfix(loss=losses.avg)

            tq.close()

        return losses.avg, final_preds

    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []

        with torch.no_grad():
            tq = tqdm(data_loader, total=len(data_loader))
            for data in tq:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                predictions, _ = self.model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions