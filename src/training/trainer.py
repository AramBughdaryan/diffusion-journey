# Class trainer.
#  Must have _load_snapshot(to load pretrained model)
# _run_batch() Must run model ob batch of data. takes input and targets as input. does backward and opt.step in that function
# _run_epoch() Must run epoch 
# _save_snapshot save model state_dict 
# train function which must be called from train.py file.
# Must initialize dataloader in init function which takes dataset, batch_size etc.
# Implement loss function.
class Trainer:
    def __init__(self) -> None:
        self.optimizer = None
        self.model = None
        pass

    def _run_batch(self, input, target):
        # run using debugger (import pdb; pdb.set_trace())
        # Check that before loss.backward() there are no gradients
        # and after that they pop up
        # Check that weights are updated after self.optimizer.step()

        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss_fn() # What loss should be used? MSE? KL_Divergence?!
        loss.backward()
        self.optimizer.step()

