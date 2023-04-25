import logging
import os
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for batch_x, batch_y, index in tqdm(train_loader):
        x_d = batch_x.cuda()
        labels = batch_y
        labels = labels.type(torch.FloatTensor).cuda()

        pred_d = net(x_d)

        optimizer.zero_grad()
        
        loss = criterion(pred_d, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = metrics.mean_squared_error(np.squeeze(
        pred_epoch), np.squeeze(labels_epoch)) ** 0.5

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ RMSE:{:.4}'.format(
        epoch + 1, ret_loss, rho_s, rho_p, rmse))

    return ret_loss, rho_s, rho_p, rmse


def eval_epoch(epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for batch_x, batch_y , index in tqdm(test_loader):
            pred = 0
           
            x_d = batch_x.cuda()
            labels = batch_y
            labels = labels.type(torch.FloatTensor).cuda()
            pred = net(x_d)
            
            # compute loss
            loss = criterion(pred, labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = metrics.mean_squared_error(np.squeeze(
            pred_epoch), np.squeeze(labels_epoch)) ** 0.5

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(
            epoch + 1, np.mean(losses), rho_s, rho_p, rmse))
        return np.mean(losses), rho_s, rho_p, rmse
    
    
def train_epoch_hall(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for batch_x, batch_est_x, batch_y, index in tqdm(train_loader):
        x_d = batch_x.cuda()
        x_est_d = batch_est_x.cuda()
        labels = batch_y
        labels = labels.type(torch.FloatTensor).cuda()

        pred_d = net(x_d,x_est_d)

        optimizer.zero_grad()
        
        loss = criterion(pred_d, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = metrics.mean_squared_error(np.squeeze(
        pred_epoch), np.squeeze(labels_epoch)) ** 0.5

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ RMSE:{:.4}'.format(
        epoch + 1, ret_loss, rho_s, rho_p, rmse))

    return ret_loss, rho_s, rho_p, rmse


def eval_epoch_hall(epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for batch_x,batch_est_x, batch_y , index in tqdm(test_loader):
            pred = 0
           
            x_d = batch_x.cuda()
            x_est_d = batch_est_x.cuda()
            labels = batch_y
            labels = labels.type(torch.FloatTensor).cuda()
            pred = net(x_d,x_est_d)
            
            # compute loss
            loss = criterion(pred, labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = metrics.mean_squared_error(np.squeeze(
            pred_epoch), np.squeeze(labels_epoch)) ** 0.5

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(
            epoch + 1, np.mean(losses), rho_s, rho_p, rmse))
        return np.mean(losses), rho_s, rho_p, rmse
    
def train(net, criterion, optimizer, scheduler, train_loader, test_loader, args):
    
    model_tag = 'bs_{}_seed_{}_{}_hall_{}_{}_{}'.format(args['batch_size'],args['seed'],args['loss_type'],args['hallucinate'],args['att_method'],args['apply_att_method'])
    
    # create log and tensorboard directory ------------------------------------
    log_path = os.path.join(args['output_dir'], args['dataset'], 'logs')
    tensorboard_path = os.path.join(args['output_dir'], args['dataset'], 'tensorboard')
    if not os.path.exists(log_path):
        print('Creating log directory: {:s}'.format(log_path))
        os.makedirs(log_path)
    if not os.path.exists(tensorboard_path):
        print('Creating tensorboard directory: {:s}'.format(tensorboard_path))
        os.makedirs(tensorboard_path)
    
    # create model directory ---------------------------------------------------
    model_path = os.path.join(args['output_dir'], args['dataset'], 'models/', model_tag)
    if not os.path.exists(model_path):
        print('Creating model directory: {:s}'.format(model_path))
        os.makedirs(model_path)
        
    # set logger ----------------------------------------------------------------
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(
        log_path, model_tag) + '.log', level=logging.INFO, format=LOG_FORMAT)
    
    # set tensorboard -----------------------------------------------------------
    writer = SummaryWriter(tensorboard_path)
    
    # set best results ---------------------------------------------------------
    best_srocc = 0
    best_plcc = 0
    besst_rmse = 1
    
    # start training ------------------------------------------------------------
    print('Now starting training for {:d} epochs'.format(args['n-epochs']))
    for epoch in range(args['n-epochs']):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        if not args['hallucinate']:
            loss_train, rho_s, rho_p, rmse = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)
        else:
            loss_train, rho_s, rho_p, rmse = train_epoch_hall(epoch, net, criterion, optimizer, scheduler, train_loader)
       
        writer.add_scalar('train/loss', loss_train, epoch)
        writer.add_scalar('train/rho_s', rho_s, epoch)
        writer.add_scalar('train/rho_p', rho_p, epoch)
        writer.add_scalar('train/rmse', rmse, epoch)
        
        logging.info('Starting eval...')
        logging.info('Running testing in epoch {}'.format(epoch + 1))
        if not args['hallucinate']:
            loss, rho_s, rho_p, rmse = eval_epoch(epoch, net, criterion, test_loader)
        else:
            loss, rho_s, rho_p, rmse = eval_epoch_hall(epoch, net, criterion, test_loader)
        print('Eval model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(
                epoch + 1, rho_s, rho_p, rmse))
        logging.info('Eval done...')
        
        # save best results -----------------------------------------------------
        if rho_s > best_srocc or rho_p > best_plcc or rmse < besst_rmse:
            best_srocc = rho_s
            best_plcc = rho_p
            besst_rmse = rmse
            logging.info('Best weights and model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(
                epoch + 1, best_srocc, best_plcc, besst_rmse))
            
        torch.save(net, model_path + 'epoch_{}.pth'.format(epoch + 1))
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
        
        

    