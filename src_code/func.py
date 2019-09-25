import numpy as np
from gensim.models import Word2Vec
import random
import json
from sklearn.manifold import TSNE
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def wl (dd,t):
    g=np.load('cat_'+str(t)+'.npy')
    D=dd
    label=np.zeros((g.shape[0], D+1))

    newlabel=''

    for aa in range(0, label.shape[0]):
        neighbor = []
        nei_label = []
        for ab in range(0, label.shape[0]):
            if g[aa][ab] != 0:
                neighbor.append(ab)
                nei_label.append(g[aa][ab])
                
        for d in range(1,D+1):
            newlabel=newlabel+str(label[aa][d-1])+':'
            ne=[]
            for n in range(0,len(neighbor)-1):
                ne.append(str(label[neighbor[n]][d-1])+str(nei_label[n]))
            ne.sort()
            for n in range(0,len(neighbor)-1):
                newlabel=newlabel+ne[n]
            label[aa][d]=str(hash(newlabel))
            newlabel=''
                            
    np.save("label_"+str(dd)+"_"+str(t), label)
    
    subgraph(dd,t)
    
def subgraph(dd,t):
    g=np.load('cat_'+str(t)+'.npy')
    label=np.load("label_"+str(dd)+"_"+str(t)+".npy")

    subgraph=[]
    D=dd
    for aa in range(0, label.shape[0]):
        neighbor = []
        sub=[]
        for ab in range(0, label.shape[0]):
            if g[aa][ab] != 0:
                neighbor.append(ab)
        for d in range(0, D+1):
            if d == 0:
                sub.append(str(label[aa][d]))
                sub.append(str(label[aa][d+1]))
                for i in range(0, len(neighbor)-1):
                    sub.append(str(label[neighbor[i]][d]))
                    sub.append(str(label[neighbor[i]][d+1]))
                
            elif d == D:
                sub.append(str(label[aa][d]))
                sub.append(str(label[aa][d-1]))
                for i in range(0, len(neighbor)-1):
                    sub.append(str(label[neighbor[i]][d]))
                    sub.append(str(label[neighbor[i]][d-1]))
                
            else :
                sub.append(str(label[aa][d]))
                sub.append(str(label[aa][d-1]))
                sub.append(str(label[aa][d+1]))
                for i in range(0, len(neighbor)-1):
                    sub.append(str(label[neighbor[i]][d]))
                    sub.append(str(label[neighbor[i]][d-1]))
                    sub.append(str(label[neighbor[i]][d+1]))
            
        subgraph.append(sub)
        
    

    neg = 200 #num of negative samples for skipgram model -TUNE ACCORDING TO YOUR EXPERIMENTAL SETTING
    sg = 1
    embedding_dim=256
    n_cpus=4
    iters=20
    random.shuffle(subgraph)
    model = Word2Vec(subgraph, 
                 min_count=1,
                 size=embedding_dim,
                 sg=sg, #make sure this is ALWAYS 1, else cbow model will be used instead of skip gram
                 negative=neg,
                 workers=n_cpus,
                 iter=iters)


    model.train(subgraph, total_examples=len(subgraph), epochs=10)


    model.save('f256n200col2Vec_'+str(dd)+'_'+str(t)+'.model')
    model.wv.save('f256n200colvectors_'+str(dd)+'_'+str(t)+'.kv')

    ListOfSubgraphs = list(model.wv.vocab)
    DictOfVectors ={}
    for char in ListOfSubgraphs:
        colVector = model.wv[char].tolist()
        DictOfVectors[char] = colVector
    
    with open('f256n200CollaboraionVectors_'+str(dd)+'_'+str(t)+'.txt', 'w') as f:
        json.dump(DictOfVectors, f, ensure_ascii=False)
        
    tSNE(dd,t)


def tSNE(dd,t):
    label=np.load('label_'+str(dd)+'_'+str(t)+'.npy')
    with open('f256n200CollaboraionVectors_'+str(dd)+'_'+str(t)+'.txt') as data_file: 
        colVectors_json = json.loads(data_file.read())
        D=dd
        scholarVecs=[]


    colVectors = {}
    for title in colVectors_json.keys():
        colVectors[title] = np.asarray(colVectors_json[title])
   
    for aa in range(0, label.shape[0]):
        con=np.zeros((0,0))
        for d in range(1, D+1):
            con=np.concatenate((con, colVectors[str(label[aa][d])]), axis=None)
        scholarVecs.append(con)
    
 
    #Col_Vectors = list(colVectors.values())
    #Col_Tags = list(colVectors.keys())

    tsne_model = TSNE(perplexity=45, n_components=2, init='pca', n_iter=1000, random_state=23, learning_rate=800)
    new_values = tsne_model.fit_transform(scholarVecs)

    cite_cr = np.load('cite_cr.npy')
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    fig=plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        #plt.scatter(x[i],y[i])
        #논문편수 1개만 빨간색
        
        if cite_cr[i][1]==1 :
            plt.scatter(x[i],y[i],color='gray')
        elif cite_cr[i][1]==2:
            plt.scatter(x[i],y[i],color='black')
        elif cite_cr[i][1]==3:
            plt.scatter(x[i],y[i],color='blue')
        elif cite_cr[i][1]==4:
            plt.scatter(x[i],y[i],color='green')
        elif cite_cr[i][1]==5:
            plt.scatter(x[i],y[i],color='yellow')
        elif (cite_cr[i][1]>=6) and (cite_cr[i][1]<=13):
            plt.scatter(x[i],y[i],color='orange')
        else:
            plt.scatter(x[i],y[i],color='red')
        
    fig.savefig('f256n200fig'+str(dd)+'_'+str(t)+'.png', dpi=300)
        
        
def vector_similarity(dd,t):
    label=np.load('label_'+str(dd)+'_'+str(t)+'.npy')
    with open('f256n200CollaboraionVectors_'+str(dd)+'_'+str(t)+'.txt') as data_file: 
        colVectors_json = json.loads(data_file.read())
        D=dd
        scholarVecs=[]


    colVectors = {}
    for title in colVectors_json.keys():
        colVectors[title] = np.asarray(colVectors_json[title])
   
    for aa in range(0, label.shape[0]):
        con=np.zeros((0,0))
        for d in range(1, D+1):
            con=np.concatenate((con, colVectors[str(label[aa][d])]), axis=None)
        scholarVecs.append(con)
        
    sim_cos=np.zeros((label.shape[0],label.shape[0]))
    sim_l1=np.zeros((label.shape[0],label.shape[0]))
    sim_l2=np.zeros((label.shape[0],label.shape[0]))
    
    for aa in range(0, label.shape[0]):
        vA = scholarVecs[aa]
        for ab in range(0, label.shape[0]):
            vB=scholarVecs[ab]
            sim_cos[aa][ab]=np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))
            sim_l1[aa][ab]=1/(np.linalg.norm(vA-vB, 1)+1)
            sim_l2[aa][ab]=1/(np.linalg.norm(vA-vB)+1)
        
    np.save("f256n200simcos_"+str(dd)+"_"+str(t), sim_cos)
    np.save("f256n200siml1_"+str(dd)+"_"+str(t), sim_l1)        
    np.save("f256n200siml2_"+str(dd)+"_"+str(t), sim_l2)  
    
    
        
dl=(3,4)
#dl=(3,4)
#thetal=(0.25)
#thetal=(0.25,0.5)
'''
for d in dl:
    for t in thetal:
        #vector_similarity(d, t)
        subgraph(d,t)
        #tSNE(d,t)
'''        

#subgraph(3,0.25)
vector_similarity(3,0.25)
        
        
        
        
        
        
        
    