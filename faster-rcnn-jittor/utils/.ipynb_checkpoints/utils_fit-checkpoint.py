import os

import jittor as jt
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            # print(f"box的形状{jt.array(boxes).numpy.shape}") # 114
            # with jt.no_grad():
            #     if cuda:
            #         images = images.cuda()

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler)
            jt.gc() 
            total_loss      += total.item()
            rpn_loc_loss    += rpn_loc.item()
            rpn_cls_loss    += rpn_cls.item()
            roi_loc_loss    += roi_loc.item()
            roi_cls_loss    += roi_cls.item()
            
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'rpn_loc'       : rpn_loc_loss / (iteration + 1),  
                                'rpn_cls'       : rpn_cls_loss / (iteration + 1), 
                                'roi_loc'       : roi_loc_loss / (iteration + 1), 
                                'roi_cls'       : roi_cls_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with jt.no_grad():
                if cuda:
                    images = images.cuda()

                train_util.optimizer.zero_grad()
                _, _, _, _, val_total = train_util.execute(images, boxes, labels, 1)
                val_loss += val_total.item()
                
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        jt.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        jt.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
    jt.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    jt.gc() 