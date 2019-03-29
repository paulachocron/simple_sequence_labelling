def get_data(train,val,test,dest_path, mode):
    dest_train = dest_path+'train'
    dest_dev = dest_path+'val'
    dest_test = dest_path+'test'
    # dest_test = 'data/romanian_clean/ro.dev'

    for src, dest in [(train,dest_train),(dev,dest_dev),(test,dest_test)]:
        with open(dest, 'w+') as d, open(src, encoding="UTF-8") as f:
            for line in f:
                line = line.rstrip()
                if len(line) == 0:
                    d.write("{}\n".format(line))

                elif line[0] == '#':
                    continue        
                else:
                    _, word, _, pos, _, _, _, _,_,_ = line.split()
                    word = word.lower() # this is to make the dataset more twitter-friendly
                    if mode=='pos':
                        d.write("{} {}\n".format(word, pos))
                    elif mode=='ner':
                        if pos=='PROPN':
                            d.write("{} {}\n".format(word, pos))
                        else:
                            d.write("{} O\n".format(word))
                

train = 'data/UD_Spanish-AnCora-master/es_ancora-ud-train.conllu' 
dev = 'data/UD_Spanish-AnCora-master/es_ancora-ud-dev.conllu' 
test = 'data/UD_Spanish-AnCora-master/es_ancora-ud-test.conllu' 

dest_path = 'data/es_pos/'

get_data(train,dev,test,dest_path, 'pos')
