    
    # calculate how many terms exist. sample this number
    lo_term = pd.unique(ds.term)
    
    # randomly sample the words with replacment
    #indexces = np.sort(choices(range(0, len(lo_term)), k=len(lo_term)))
    #count for each word, the number of it repetitions
    num_boot_iter = 100
    statistics = []
    HC_stat = []
    
    #perform bootstrap iterations
    for b_i in range(0, num_boot_iter):
        print(b_i)
        k = len(lo_term) # take all unique words and randmply sample k from them with replacment
        indexces = np.sort(choices(range(0, k), k=k))
    
        indexces_hist = pd.Series(indexces).value_counts().reset_index()\
                          .values.tolist() # create a histogram of the words
        newsturct = []

        # for each term in the histogram, create a coresponding configuration of thers
        # replace repeated terms with "Term_2", where the last one is a running number
        for i in range(0,len(indexces_hist)):
            c_i = indexces_hist[i]
            
            #filter all that match c_i[0][0], and rename them to "Term_2",
            for t in range(0, c_i[1]):
                ds_auth = ds[ds.term == lo_term[c_i[0]]]
                ppp = pd.DataFrame(ds_auth.loc[:, 'term'].tolist())
                ppp1 = str(t) +  '_'  + ppp[1] 
                ds_auth.loc[:, 'term'] = list(zip(ppp[0], ppp1))
                
                #concatinate to the data structure
                if i==0:
                    newsturct =  ds_auth
                else:
                    newsturct = pd.concat([newsturct, ds_auth]) 
                    
                    
        
        # collect statistics: doc_id, wrt_author, HC - list
        newsturct = newsturct.sort_values(by=['doc_id'])  

        # create a model with the new words dictionary              
        model = AuthorshipAttributionDTM(newsturct, min_cnt = 5, randomize = False)
    
        #compute scores (each doc against each corpus)
        df = model.internal_stats(LOO = False) #use LOO = true only for rank-based testing
    
        d2 = np.asarray(df['HC'].tolist())
        d3 = d2.reshape(len(d2), 1)       
    
               
        if b_i==0:
            HC_stat = d3
        else:
            if len(d3)==len(HC_stat):
                HC_stat = np.concatenate((HC_stat, d3),axis=1)
                
        #hc_stat = run _hc(newsturct)
    'doc_id', 'wrt_author', 
    alpha = 0.8
    lower = []
    upper = []
    for i in range(0, len(HC_stat)):
        current_comparison = HC_stat[i,:];
        ordered = np.sort(current_comparison)
        lower.append(np.percentile(ordered, 100*(1-alpha)/2))
        upper.append(np.percentile(ordered, 100*(alpha+((1-alpha)/2))) )
    lower = np.asarray(lower).reshape(len(lower), 1)      
    upper = np.asarray(upper).reshape(len(upper), 1)      
    
    c = np.concatenate((lower, upper),axis=1)
    np.savetxt(r"C:\Users\golovin\Google Drive\Phd\Articles\NLP_Bible\Code\Authorship\Authorship\pythonCodeV2\drive\My Drive\Data\HC_lower_uperbounts.csv", c, delimiter=",")
    df.to_csv(r"C:\Users\golovin\Google Drive\Phd\Articles\NLP_Bible\Code\Authorship\Authorship\results_python\HC_results_boot_stap.csv")
    
    np.savetxt(r"C:\Users\golovin\Google Drive\Phd\Articles\NLP_Bible\Code\Authorship\Authorship\pythonCodeV2\drive\My Drive\Data\HC_stat_100.csv", HC_stat, delimiter=",")
