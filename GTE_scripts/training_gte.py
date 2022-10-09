def train(model, vocab,
          train_data,
          optimizer,
          criterion, 
          device,
          train_history=None, valid_history=None):
    
    model.train()
    iteration, num_batches = 0, 0
    temp_loss, epoch_loss = 0, 0
    history = []

    for batch_dictionary in generate_batches(train_data, vocab, device = device):  
        
        iteration += 1
        num_batches += 1
        # reset gradients
        model.zero_grad()
                
        prem, hypo, len_prem, len_hypo = batch_dictionary.values()
        # compute output 
        log_probs, _ = model(prem, hypo, len_prem, len_hypo)
        num_classes = log_probs.size(-1) 
        
        # compute loss for a batch
        batch_loss = criterion(log_probs.view(-1, num_classes), 
                               hypo[:,1:].contiguous().view(-1))
        temp_loss += batch_loss.item()
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # update parameters
        optimizer.step()
        
        # logging and reporting
        history.append(temp_loss / iteration)
        
        if num_batches % 100 == 0:
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            clear_output(True)
            print( temp_loss / iteration )
            temp_loss, iteration = 0, 0
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='training history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='validation history')
            plt.legend()
            plt.show()
            
    return epoch_loss / num_batches   


def evaluate(model, vocab, dev_data, criterion, device):
    
    model.eval()
    epoch_loss, num_batches = 0, 0
    
    with torch.no_grad():
    
        for batch_dictionary in tqdm(generate_batches(dev_data, vocab, device = device)):  
        
            num_batches += 1
            
            # reset gradients
            model.zero_grad()
                    
            # compute probability distribution over vocabulary
            prem, hypo, len_prem, len_hypo = batch_dictionary.values()
            log_probs, _ = model(prem, hypo, len_prem, len_hypo)
            num_classes = log_probs.size(-1) 
            
            # compute loss for a batch
            batch_loss = criterion(log_probs.view(-1, num_classes), 
                               hypo[:,1:].contiguous().view(-1))
            epoch_loss += batch_loss.item()            
    
        return epoch_loss / num_batches



def predict(premise, model, device, max_len=20):
    
    input_word = torch.tensor([voc.vocabulary["<sos>"]]).view(1, -1).to(device)
    prem = voc.sentence2tensor(premise).view(1, -1).to(device)
    len_prem = torch.tensor([prem.size(1)]).to(device)
    
    model.eval()
    
    with torch.no_grad():
        # embed premise
        emb_prem = model.embedding(prem)
        mask_prem = torch.ne(prem, .0)
        
        # retrieve contextual representation of premise(annotations) and the final hidden state
        enc_out, enc_state = model.encoder(emb_prem, len_prem)
        
        # generate the hypothesis and compute attention weights
        entailed_sentence, attns = [], []
        dec_state = enc_state
        hypo_len = torch.tensor([1]).to(device)
        for step in range(max_len):
            
            dec_input = model.embedding(input_word)
            dec_out, dec_state, attn =\
            model.decoder(dec_input, enc_out, dec_state, mask_prem, hypo_len=hypo_len)
            attns.append(attn.squeeze())
            
            # greedy decoding 
            _, topi = dec_out.squeeze().data.topk(1)
            topi = topi.view(-1)
            decoded_word = voc.vocabulary.lookup_tokens([int(topi)])[0]
            if decoded_word == "<eos>":
                break
    
            input_word = topi.detach().view(-1, 1)
            entailed_sentence.append(decoded_word)
      
        attns = torch.stack(attns).squeeze().cpu().numpy()
        
    return " ".join(entailed_sentence), attns