import torch
from torch.utils.data import DataLoader
# import sys
# sys.path.append('model/')
from model import Demoire
from dataset import Custom_Dataset
from losses import KTLoss
import os
import wandb
import pyiqa
import tqdm
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


obj = time.localtime()
code_start_time = time.asctime(obj)

class Train():
    def __init__(self,args) -> None:
        self.args = args
        self.current_epoch = 0
        self.best_psnr_student = 0
        self.best_ssim_student = 0
        self.best_psnr_teacher = 0
        self.best_ssim_teacher = 0
        self.model_intiliaser()
        self.data_intiliaser()
        self.losses_opt_and_metrics_init()
        self.init_summary()
    def model_intiliaser(self):
        if self.args.model_type.endswith("CR"):
            model_name = self.args.model_type.split('_')[0]
        else:
            model_name = self.args.model_type
        print(f'Model Name = {model_name}')
        if model_name=='KTDN':
            self.teach_model = Demoire()
            self.student_model  = Demoire()
        else:
            print("Enter a valid model name")
        self.teach_model = self.teach_model.to(self.args.device)
        self.student_model = self.student_model.to(self.args.device)
        return None
    def data_intiliaser(self):
        train_ds = Custom_Dataset(self.args.train_path,im_shape = self.args.im_shape ,is_train=True)
        self.train_dataloader = DataLoader(train_ds,batch_size=self.args.batch_size,shuffle=True,num_workers=8)
        val_ds = Custom_Dataset(self.args.test_path,is_train=False)
        self.val_dataloader = DataLoader(val_ds,batch_size=20,shuffle=False,num_workers=8)
        return None
    def init_summary(self):
        wandb.init(project=f"{self.args.model_type}",name=f"{self.args.model_type}_{code_start_time}")
        return
    def losses_opt_and_metrics_init(self):
        total_count = self.args.epochs*len(self.train_dataloader)
        self.optimizer_teacher = torch.optim.Adam(self.teach_model.parameters(), lr=self.args.LR)
        self.optimizer_student = torch.optim.Adam(self.student_model.parameters(), lr=self.args.LR)
        self.scheduler_teacher = CosineAnnealingLR(self.optimizer_teacher,total_count,self.args.LR*(10**(-4)))
        self.scheduler_student = CosineAnnealingLR(self.optimizer_student,total_count,self.args.LR*(10**(-4)))
        # if self.args.model_type.endswith('CR'):
        #     self.criterion_CR = ContrastLoss(self.args.device)
        self.criterion_l1_teacher = torch.nn.L1Loss().to(self.args.device)
        self.criterion_l1_student = torch.nn.L1Loss().to(self.args.device)
        self.criterion_kt = KTLoss(alpha=1)
        self.psnr = pyiqa.create_metric('psnr').to(self.args.device)
        self.ssim = pyiqa.create_metric('ssim').to(self.args.device)

    def train_epoch(self):
        self.teach_model.train()
        self.student_model.train()
        for count,(inputs, gt) in enumerate(tqdm.tqdm(self.train_dataloader)):
            inputs = inputs.to(self.args.device)
            gt = gt.to(self.args.device)
            self.optimizer_teacher.zero_grad()
            self.optimizer_student.zero_grad()
            with torch.set_grad_enabled(True):
                student_outputs,student_features = self.student_model(inputs)
                teacher_outputs,teacher_features = self.teach_model(gt)
                loss_teacher = self.criterion_l1_teacher(teacher_outputs,gt)
                loss_student = self.criterion_l1_student(student_outputs,gt)
                loss_kt = self.criterion_kt(teacher_features.detach(),student_features)
                loss_stu = loss_student+loss_kt
                # if self.args.model_type.endswith('CR'):
                #     loss+=0.1*self.criterion_CR(outputs,gt,inputs)
                loss_teacher.backward()
                self.optimizer_teacher.step()
                self.scheduler_teacher.step()
                loss_stu.backward()
                self.optimizer_student.step()                
                self.scheduler_student.step()
        wandb.log({'train_l1_loss_teacher':loss_teacher.item()})
        wandb.log({'train_l1_loss_student':loss_student.item()})
        wandb.log({'train_l1_loss_kt':loss_kt.item()})
        wandb.log({'Learning rate':self.optimizer_teacher.param_groups[0]['lr']})
        return None
    def save_checkpoint(self,type='last'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder,self.args.model_type)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder,f'{type}.pth')
        save_data = {
            'step': self.current_epoch,
            f'best_psnr_student':self.best_psnr_student,
            f'best_ssim_student':self.best_ssim_student,
            f'best_psnr_teacher':self.best_psnr_teacher,
            f'best_ssim_teacher':self.best_ssim_teacher,
            'teacher_state_dict': self.teach_model.state_dict(),
            'student_state_dict': self.student_model.state_dict(),
            'teacher_optimizer_state_dict': self.optimizer_teacher.state_dict(),
            'student_optimizer_state_dict': self.optimizer_student.state_dict(),
            'student_scheduler_state_dict': self.scheduler_student.state_dict(),
            'teacher_scheduler_state_dict': self.scheduler_teacher.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)

    def load_model_checkpoint_for_training(self,type ='best_student'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder,self.args.model_type)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.current_epoch = data['step']
        self.best_psnr_student = data['best_psnr_student']
        self.best_ssim_student = data['best_ssim_student']
        self.best_psnr_teacher = data['best_psnr_teacher']
        self.best_ssim_teacher = data['best_ssim_teacher']
        self.teach_model.load_state_dict(data['teacher_state_dict'])
        self.student_model.load_state_dict(data['student_state_dict'])
        self.optimizer_teacher.load_state_dict(data['teacher_optimizer_state_dict'])
        self.optimizer_student.load_state_dict(data['student_optimizer_state_dict'])
        self.scheduler_teacher.load_state_dict(data['teacher_scheduler_state_dict'])
        self.scheduler_student.load_state_dict(data['student_scheduler_state_dict'])
        print(f"Restored model at epoch {self.current_epoch}.")

    def val_epoch(self):
        self.teach_model.eval()
        self.student_model.eval()
        psnr_value_student = []
        psnr_value_teacher = []
        ssim_value_student = []
        ssim_value_teacher = []
        for inputs, gt in tqdm.tqdm(self.val_dataloader):
            inputs = inputs.to(self.args.device)
            gt = gt.to(self.args.device)
            with torch.set_grad_enabled(False):
                outputs_teacher,_ = self.teach_model(gt)
                outputs_student,_ = self.student_model(inputs)
                # _ = self.criterion(outputs,gt)
            psnr_value_student.append(self.psnr(outputs_student,gt).mean().item())
            psnr_value_teacher.append(self.psnr(outputs_teacher,gt).mean().item())
            ssim_value_student.append(self.ssim(outputs_student,gt).mean().item())
            ssim_value_teacher.append(self.ssim(outputs_teacher,gt).mean().item())
        wandb.log({'val_psnr_student':np.mean(psnr_value_student)})
        wandb.log({'val_ssim_student':np.mean(ssim_value_student)})
        wandb.log({'val_psnr_teacher':np.mean(psnr_value_teacher)})
        wandb.log({'val_ssim_teacher':np.mean(ssim_value_teacher)})
        val_psnr_student = np.mean(psnr_value_student)
        val_ssim_student = np.mean(ssim_value_student)
        val_psnr_teacher = np.mean(psnr_value_teacher)
        val_ssim_teacher = np.mean(ssim_value_teacher)

        if val_psnr_student>self.best_psnr_student:
            self.best_psnr_student = val_psnr_student
            self.save_checkpoint('best_student')
        if val_ssim_student>self.best_ssim_student:
            self.best_ssim_student = val_ssim_student
        if val_psnr_teacher>self.best_psnr_teacher:
            self.best_psnr_teacher = val_psnr_teacher
            self.save_checkpoint('best_teacher')
        if val_ssim_teacher>self.best_ssim_teacher:
            self.best_ssim_teacher = val_ssim_teacher

        self.save_checkpoint('last')
        current_lr = self.optimizer_student.param_groups[0]['lr']
        print(f'Epoch = {self.current_epoch} Val student best PSNR = {self.best_psnr_student},Val student current PSNR = {val_psnr_student},Val best student SSIM = {self.best_ssim_student},Val student current SSIM = {val_ssim_student}')
        print(f'Epoch = {self.current_epoch} Val teacher best PSNR = {self.best_psnr_teacher},Val teacher current PSNR = {val_psnr_teacher},Val best teacher  SSIM = {self.best_ssim_teacher},Val teacher current SSIM = {val_ssim_teacher}, lr ={current_lr}')
        return None
    def run(self):
        self.load_model_checkpoint_for_training()
        for epoch in range(self.current_epoch,self.args.epochs):
            self.current_epoch = epoch
            self.train_epoch()
            if epoch%10==0:
                self.val_epoch()
            # self.val_epoch()
        return None



