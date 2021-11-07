############ Bagging- Accuracy evaluation ##########
acc_sum = 0
acc = []
corpus_size = [14,14,22]  #8,8,8)#
max_n_p = 0.75
max_n_prim_p = 0.8
max_n =  [round(i * max_n_p) for i in corpus_size]
max_n_prim = [round(i *max_n_prim_p ) for i in max_n]
lo_corp = [1, 2, 3]
lo_corp_names = ['Dtr', 'DtrH', 'P']
t_p = 1
  

print(max_n)
print(max_n_prim)
import itertools
import numpy
accuracy_all = list();



# perform bagging several times
# each time create a random train test corpora, with proportion 0.75, 0.25
#then sample the test 0.8, 0.2, and calculate accuracy score

for ttt in range(1,10):   
    cor_all = []

    train_inx_ids = list()
    test_inx_ids = list()
    # for each of the 3 corpora, sample equal proportional number of texts
    for c in lo_corp:
      i = c-1
      # select the current corpus
      current_corpus = ds[ds.author==lo_corp_names[i]]
      chapters = pd.unique(current_corpus.doc_id)
      
      rand_inx = numpy.random.permutation(corpus_size[i])
      
      train_inx = rand_inx[0:max_n[i]]
      test_inx  = rand_inx[max_n[i]+1:corpus_size[i]]
      test_inx  = test_inx[0:(corpus_size[i]-max_n[i])]
      
      train_inx_id = chapters[train_inx]
      test_inx_id = chapters[test_inx]
      
      if i==0:
          train_inx_ids = list(train_inx_id)
          test_inx_ids =  list(test_inx_id)
      else:
          train_inx_ids = list(itertools.chain(train_inx_ids, list(train_inx_id)))
          test_inx_ids = list(itertools.chain(test_inx_ids, list(test_inx_id))) 


    
    print(train_inx_ids)
    print('**************')
    print(test_inx_ids)
  
    
   
    ######## Phase 2, sample again and run the model
    
    #sample of each writer, with replacment
    
    iteration_num = 10
    t_ids = test_inx_ids
      
    for i in range(0, iteration_num):
      accuracy=list()
      exact = 0
      q = 1
      print(i)
      
      for t in range(0,len(t_ids)): #length(t_ids)
        d = t_ids[t]
        #print(d)
        train_inx_ids_2 = list()
        j = 0
        score_wrt_corpus = []
        
        for k in lo_corp:
          c = k-1
      
          train_current_doc_names = train_inx_ids[j:(j+max_n[c])]
    
          j = j+max_n[c]
          
          train_inx_2 = numpy.random.permutation(max_n[c])
          train_inx_2 = train_inx_2[0:max_n_prim[c]].astype(int)
          
          
          train_inx_id_2 = [train_current_doc_names[k] for k in train_inx_2] 
          
          train_inx_ids_2 = list(itertools.chain(train_inx_ids_2, list(train_inx_id_2))) 
          
             
              
        filtered_data = list();
                
        ids = train_inx_ids_2
        for hh in range(0, len(ids)):
            print( len(ds[ds.doc_id==ids[hh]]))
            if hh==0:
                filtered_data = ds[ds.doc_id==ids[hh]]
            else:
                filtered_data = filtered_data.append(ds[ds.doc_id==ids[hh]])
        filtered_data = filtered_data.append(ds[ds.doc_id==d])
               
        #run the model          
        newsturct = filtered_data.sort_values(by=['doc_id'])                
        model = AuthorshipAttributionDTM(newsturct, min_cnt = 5, randomize = False)
      
        #compute scores (each doc against each corpus)
        df = model.internal_stats(LOO = False) #use LOO = true only for rank-based testing
        tested_doc = df[df.doc_id==d]
        min_index = np.argmin(tested_doc.HC)
        authers = tested_doc.wrt_author
        predicted_corpus  = authers.values[min_index]
    
    
        expected_author = tested_doc.author
        expected_author  = expected_author.values[0]
        accuracy.append( predicted_corpus==expected_author)
    
      accuracy_all.append(sum(accuracy)/len(accuracy))

     
print(accuracy_all)
print(np.median(accuracy_all))

