from Dataset.test_datasets import fixed_train_video_loader, fixed_val_video_loader
from Train_video.Train import train_video_model
from LoraFBSNet_Model.Lora_model import lorafb_snet18
import torch
model = lorafb_snet18(num_classes=3,
                      cnf="ADD",
                      local_recurrence=True,
                      recurrent_scale=1.0,
                      feedback_range=3)
trained_model, history = train_video_model(model=model,
                                          train_loader=fixed_train_video_loader,
                                          val_loader=fixed_val_video_loader,
                                          epochs=100,
                                          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                          overfit_patience=3,
                                          patience=5)
