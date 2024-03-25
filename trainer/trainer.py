def train_epoch(train_dataloader,net,loss_mse,optimizer,prop,para,current_epoch):
    
    for batch,(x,y) in (enumerate(train_dataloader)):
            optimizer.zero_grad()
            pred_y = net(x)            

            measured_y = prop(pred_y[0, 0, :, :],dist=para.dist)
            loss_mse_value = loss_mse(y.float(),measured_y.float())
            loss_value =  loss_mse_value

            # backward proapation

            loss_value.backward()
            optimizer.step()
            
            # scheduler.step() 
            
            step = current_epoch * len(train_dataloader) + batch
            
            
            
