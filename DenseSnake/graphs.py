import matplotlib.pyplot as plt
import numpy as np

data = []

data_score1 = [ 1.  ,        3.      ,    1.       ,   3.     ,     1.    ,      2.,
   0.  ,        5.    ,      4.   ,       3.    ,      3.     ,     7.,
   6.   ,       4.   ,       5.    ,      3.    ,      5.   ,      13.,
   1.    ,      1.   ,       5.   ,       1.    ,      1.   ,       3.,
   4.   ,      3.   ,       9.    ,      2.    ,      6.    ,     14.,
  10.   ,      0.   ,      12.    ,      5.  ,        8.    ,      8.,
  10.   ,       7.   ,       8.  ,        6.   ,       7.   ,       4.,
   9.   ,       2.   ,       0.   ,       8.   ,      10.   ,       3.,
   0.   ,       4.   ,       4.   ,      10.   ,       5.   ,       8.,
  23.   ,      13. ,        11.      ,    7.    ,      6.  ,       11.        ]

data_reward1= [-1.50444637 , 0.18958247, -3.94247506 , 0.72852734 , 2.72129939 ,-6.1035518,
  -1.67380581 ,-4.16965005 ,-0.79488952 , 2.15923002 ,-5.59570617 ,-3.23187509,
   0.49681461 , 4.51607331 , 0.41866093 , 2.70059555 ,-3.76136994 ,-1.02542358,
  -0.46313216 ,-3.37377836 ,-1.41277567 ,-2.65080712 ,-0.70360135 ,-1.45247391,
   1.8358186 ,  2.62108533,  0.40242786 ,-2.69222031 ,-3.94392235 ,-2.92036409,
  -0.40579925, -0.81371845 , 0.8067221  , 4.96295929 ,-2.67077223 , 3.54327146,
   0.72461666,  5.30278058 ,-1.73069174 , 6.53634131 ,-1.71395126 , 0.64532253,
   1.5865009  , 4.69620116 , 0.75396822 ,-5.09075701 , 6.25226747 , 6.67208106,
   0.09372501, -2.155428  ]
data_explo = [ 0.9       ,  0.855      , 0.81225    , 0.7716375 ,  0.73305562 , 0.69640284,
   0.6615827  , 0.62850357 , 0.59707839 , 0.56722447 , 0.53886325 , 0.51192008,
   0.48632408 , 0.46200787 , 0.43890748 , 0.41696211 , 0.396114   , 0.3763083,
   0.35749289 , 0.33961824 , 0.32263733 , 0.30650546 , 0.29118019 , 0.27662118,
   0.26279012 , 0.24965062 , 0.23716809 , 0.22530968 , 0.2140442  , 0.20334199,
   0.19317489 , 0.18351614 , 0.17434034 , 0.16562332 , 0.15734215 , 0.14947505,
   0.14200129 , 0.13490123 , 0.12815617 , 0.12174836 , 0.11566094 , 0.10987789,
   0.104384  ,  0.0991648  , 0.09420656 , 0.08949623 , 0.08502142 , 0.08077035,
   0.07673183 , 0.07289524, ]

data_score2 = [ 3.     ,     4.     ,     2.     ,     0.     ,     4.      ,    4.,
   0. ,         6.  ,        2.    ,      0.     ,     1.   ,       2.,
   7.    ,      5.   ,       4.   ,       5.    ,      5.    ,      8.,
   5.    ,      5.  ,        0.     ,     7.    ,      5.    ,      3.,
   1.   ,       9.  ,        5.   ,       7.    ,      8.    ,      4.,
   0.   ,       4.  ,        0.    ,     11.    ,      6.    ,     10.,
   9.  ,        4.  ,       14.  ,        8.    ,     11.    ,      7.,
   1.   ,       0.  ,        8.   ,      10.     ,    11.    ,     12.,
   9.  ,        9.  ,       17.   ,       1.     ,     0.    ,      0.,
   0.   ,       0.  ,        0.   ,       0.       ,   0.    ,      0.        ]

data_reward2 =  [-1.81911247, -0.03013557 ,-0.35334114 ,-4.21358819, -5.40004177,  2.44703212,
  -0.74167125, -5.15021382 , 2.61374362 , 0.11750942 ,-5.24904418 ,-2.008477,
   5.78631043 ,-0.05326266 ,-2.73652832 ,-2.46878448 ,-4.63309676 , 1.85212705,
  -1.22158687 , 0.59404716 , 1.31356714, 11.94617454 , 4.70241416 ,-2.58269613,
   0.37068984, -4.26132422, -0.03481492 ,-1.26362998 , 1.69757106 , 5.38751904,
  -0.98786481 , 5.84791679, -0.54105253 ,-3.37423802 , 8.41405604 , 1.86839095,
  -2.77273334,  3.61538077, -3.59232434 , 0.70408929, -0.48922517 , 5.14423058,
  -1.10307197 , 0.56282736 , 6.49519482, 11.12951964 ,-1.41196775 , 1.89029542,
  -1.63888395 , 3.83347147]



data_score3 = [ 0.     ,     3.    ,      2.       ,   2.   ,       3.   ,       2.,
   4.    ,      4.      ,    1.     ,     0.      ,    1.      ,    4.,
   2.    ,      3.  ,        0.  ,        5.     ,     5.      ,    0.,
   2.    ,      3.     ,     1.    ,      8.    ,      3.    ,      3.,
   0.    ,      8.   ,       0.   ,       1.    ,      4.    ,     13.,
   0.    ,      2.   ,       7.   ,      13.    ,      5.    ,      0.,
   0.   ,       0.   ,       0.   ,       1.    ,      6.    ,      2.,
   7.   ,       7.   ,       7.   ,      19.    ,      8.    ,      7.,
  12.    ,      5.  ,        9.   ,       2.     ,     0.     ,     0.,
   0.    ,      0.  ,        0.    ,      0.      ,    0.    ,      0.        ]

data_reward3 = [-4.1686213  , 5.83091208 , 3.64033294  ,1.73093094  ,7.06134879, -5.87311249,
  -2.57719394, -4.0361777  , 2.71822009, -6.73111636 , 0.6978156  , 1.21989376,
  -3.1811231 , -5.44344784 , 2.83367376 , 0.60244745 ,-1.59201067 ,-1.76577899,
  -4.03999333, -1.61037608, -1.24003886 ,-3.13447741 , 1.85178894 , 8.55436883,
   1.31733936,  1.62344379 ,-1.97088112 ,-2.50118154 , 2.8278803 , -1.55011679,
   1.14169771 ,-3.57066953, -3.5674059  ,-0.16078228 ,-1.37122473 , 0.41284778,
   0.5139933 ,  0.12917841 , 0.74232544 ,-2.81316643, -2.6771986 , -2.4730833,
   0.08676606,  5.27832019 , 4.66721736 ,-7.5330145 , -2.28039109, -7.41414589,
   9.58044962, -2.88473678 ]

data_score4 =[ 1.   ,       3.  ,        4.       ,   4.        ,  3.        ,  2.,
   0.   ,       5.   ,       2.      ,    1.        ,  4.      ,    4.,
   2.    ,      3.     ,     0.     ,     7.    ,      0.   ,       3.,
   4.    ,      7.    ,      6.     ,     3.    ,      3.    ,      5.,
   7.    ,      0.    ,      1.    ,      3.    ,      3.    ,      4.,
   5.    ,      0.    ,      3.    ,     16.    ,      8.     ,    12.,
   1.   ,       0.   ,      14.    ,      5.    ,      3.    ,      7.,
   8.   ,       0.   ,       0.    ,      0.    ,     13.    ,      1.,
   5.   ,       7.   ,       0.    ,      0.    ,      0.      ,    0.,
   0.   ,       0.  ,        0.    ,      0.     ,     0.     ,    23.        ]

data_reward4 =  [ 0.19208009 , 2.67723369  ,4.89038651,  5.28206872,  2.75866997 , 1.86749861,
  -4.70985577, -1.03497728,  5.24517723 -3.67561163, -4.23688496 , 5.6795882,
  -2.39387911 , 1.23215547, -2.01737819 ,-0.04576149 , 0.08158746, -4.01704702,
  -6.45858083 ,-2.09626564 , 3.39315066 , 1.32613529 , 1.4594452 , -2.81034351,
   2.56815806 , 2.39174371, -0.48017604 , 9.65122239 ,10.88828253 ,-0.9655584,
   1.67096293 ,-1.4824469  , 1.73097983 ,-2.08156521 ,-2.7523145 ,  0.98245297,
  -3.68782909,  2.07173075 , 0.37101477 ,13.7351854 , -3.8948459 , -1.02933013,
  -4.58273516, -1.24966686, -1.90841346 ,-2.53533578,  1.58220666 ,-1.27675088,
  10.53248278 ,-0.12902143  ]
 
data_score5 =[ 3.      ,    2.      ,    4.      ,   2.     ,     3.      ,    1.,
   2.    ,      1.    ,      1.    ,      1.    ,      3.    ,      1.,
   5.   ,      3.    ,      5.     ,     0.   ,       2.      ,    9.,
   1.    ,     11.   ,       3.    ,      5.   ,       8.   ,       4.,
   7.    ,      7.   ,       3.    ,      1.    ,      4.    ,      0.,
   3.    ,      0.   ,      12.     ,     9.    ,      3.    ,      4.,
   8.    ,      3.   ,       6.    ,      8.    ,      9.   ,       9.,
  15.   ,      11.    ,      6.    ,      8.    ,     10.  ,        4.,
   6.    ,     10.    ,      5.   ,       5.    ,      0.  ,        0.,
   0.    ,      0.     ,     0.    ,      0.     ,     0.   ,       0.        ]

data_reward5 =  [ 0.21879505, -5.03414056 ,-2.01574949 ,-4.80384388,  1.98278161 , 5.42374351,
  -2.34018347 , 1.55103259, -2.33163188 , 0.52938975, -2.03550805  ,5.9356873,
  -1.00448414, -1.59138598 , 2.47723642,  1.24598618 ,-2.58098801,  6.95106159,
  -0.31675482 ,-1.13242961 , 0.82696606 ,-4.76683708 ,-4.14271199 , 3.19263281,
  -1.13816513 ,-0.16930291 ,11.6429256 ,  4.10086458 , 3.31457705, -0.87260574,
  -0.67485688 ,-0.23061668 ,-7.22085607, -4.42846629 , 4.56097452 , 5.66769715,
  -2.62186666 ,-0.48561951 ,12.20336855, -0.74629118, -4.75715596 ,-2.13776879,
  -7.71411906,  0.15209933, -1.18500201, 12.1608402  , 7.63141591 , 5.45086295,
   0.6062134 , -1.91652525 ]

Random_scores = [ 0.     ,     0. ,        3.     ,     2.   ,       1. ,         0.,
   0.    ,      2.      ,    3.    ,      0.      ,    3.      ,    3.,
   1.   ,       1.   ,       2.         , 1.   ,       1.      ,    1.,
   0.   ,       3.    ,      4.      ,    0.    ,      1.     ,    1.,
   0.    ,      1.    ,      1.     ,     3.   ,       2.    ,      1.,
   0.   ,       2.    ,      0.    ,      2.   ,       3.    ,      1.,
   0.    ,      2.   ,       3.    ,      0.   ,       2.   ,       3.,
   0.   ,       1.    ,      2.    ,      6.   ,       0.     ,     1.,
   2.     ,     1.    ,      0.    ,      0.   ,       0.   ,       0.,
   0.     ,     0.     ,     0.    ,      0.    ,      0.   ,      0.    , 
    3.0,  1.0,  0.0,  0.0,
   2.0,  2.0,  1.0,  3.0,
   3.0,  2.0,  5.0,  5.0,
   0.0,  3.0,  3.0,  6.0,
   3.0,  0.0,  0.0,  2.0,
   1.0,  1.0,  1.0,  3.0,
   2.0,  3.0,  1.0,  1.0,
   4.0,  0.0,  8.0,  0.0,
   1.0,  0.0,  3.0,  1.0,
   2.0,  1.0,  1.0,  5.0,
   1.0,  1.0,  1.0,  1.0,
   4.0,  3.0,  3.0,  1.0,
   2.0,  0.0,  0.0,  1.0,
   2.0,  3.0,  0.0,  3.0,
   3.0,  2.0,  0.0,  4.0,
   0.0,  0.0,  1.0,  0.0,
   0.0,  0.0,  1.0,  3.0,
   2.0,  2.0,  1.0,  0.0,
   0.0,  1.0,  2.0,  0.0,
   2.0,  2.0,  2.0,  0.0,
   1.0,  1.0,  4.0,  3.0,
   2.0,  0.0,  1.0,  5.0,
   1.0,  1.0,  3.0,  0.0,
   1.0,  0.0,  1.0,  4.0,
   1.0,  0.0,  0.0,  3.0,
   1.0,  2.0,  4.0,  3.0,
   5.0,  1.0,  3.0,  2.0,
   3.0,  3.0,  2.0,  3.0,
   0.0,  3.0,  4.0,  0.0,
   1.0,  4.0,  3.0,  0.0,
   2.0,  4.0,  1.0,  1.0,
   1.0,  0.0,  2.0,  1.0,
   4.0,  5.0,  5.0,  4.0,
   3.0,  4.0,  1.0,  3.0,
   2.0,  3.0,  2.0,  1.0,
   1.0,  3.0,  5.0,  1.0,
   0.0,  3.0,  1.0,  2.0,
   4.0,  1.0]   

def get_mean_random():
    mean_random = 0
    for i in range(200):
        mean_random+=Random_scores[i]
    return mean_random/200.0

def get_mean_random_noZero():
    mean_random = 0.0
    nb=0.0
    for i in range(200):
        if(Random_scores[i]!=0.0):
            mean_random+=Random_scores[i]
            nb+=1
    return mean_random/nb

def get_data_mean(a,b,c,d,e,nb):
    data = [0.0 for i in range(nb)]
    for i in range(nb):
        data[i]= (a[i]+b[i]+c[i]+d[i]+e[i])/5.0
    return data

def get_mean_trained_score(a,b,c,d,e):
    mean_random = 0
    for j in range(10):
        i=50+j
        mean_random+= (a[i]+b[i]+c[i]+d[i]+e[i])/50.0
    return mean_random

def get_mean_trained_score_noZero(a,b,c,d,e):
    mean_random = 0.0
    nb= 0.0
    fusion =[]
    for j in range(10):
        fusion.append(a[50+j])
        fusion.append(b[50+j])
        fusion.append(c[50+j])
        fusion.append(d[50+j])
        fusion.append(e[50+j])
    for i in range(50):
        if(fusion[i]!=0.0):
            mean_random+=  fusion[i]
            nb+=1.0
    return mean_random/nb

mean_random = get_mean_random()
print(mean_random)

mean_random_noZero = get_mean_random_noZero()
print(mean_random_noZero)

mean_trained_score = get_mean_trained_score(data_score1,data_score2,data_score3,data_score4,data_score5)
print(mean_trained_score)

mean_trained_score_noZero= get_mean_trained_score_noZero(data_score1,data_score2,data_score3,data_score4,data_score5)
print(mean_trained_score_noZero)

data_mean = get_data_mean(data_score1,data_score2,data_score3,data_score4,data_score5,50)

gens = [i+1 for i in range(50)]
data_random = [mean_random for i in range(50)]

def get_graph():

    """
    plt.bar(gens, data_mean, align='center', alpha=0.5)
    plt.plot(gens, data_random, c='m', label= "average score for a random-choice policy")
    plt.ylabel('Scores')
    plt.xlabel('Generations')
    plt.title('Average Evolution of score during training of 50 generations ')
    plt.legend()
    plt.show()
    """

    fig = plt.figure()
    ax= fig.add_subplot(111)
    ax.bar(gens, data_mean, align='center', alpha=0.5)
    ax.plot(gens, data_random, c='r', label= "average score for a random-choice policy")
    ax2 = ax.twinx()
    ax.set_ylabel('Scores')
    ax.set_xlabel('Generations')

    ax2.plot(gens, data_explo, c='g', label= "exploration rate")
    ax2.set_ylabel('exploration rate', c='g')

    ax.set_title('Average Evolution of scores over 5 training sessions of 50 generations ')
    fig.legend()
    plt.show()

#get_graph()

def get_random_distrib():
    max =0.0
    for i in range(200):
        if(Random_scores[i]>max):
            max = Random_scores[i]
    random_distrib= [0 for i in range(int(max+1))]
    for i in range(200):
        random_distrib[ int(Random_scores[i]) ]+=1
    for i in range(int(max+1)):
        random_distrib[i]/=200
    return random_distrib

def plot_random_distrib():

    random_distrib = get_random_distrib()
    scores_distrib = [i for i in range(len(random_distrib))]
    plt.barh(scores_distrib, random_distrib, align='center', alpha=0.5)
    means = [mean_random for i in range(len(random_distrib))]
    means_noZero = [mean_random_noZero for i in range(len(random_distrib))]
    
    plt.plot(random_distrib,means,c='r', label= "average score for the random-choice policy")
    plt.plot(random_distrib,means_noZero,c='g', label= "average score for the random-choice policy, discarding null scores")
    plt.ylabel('Proportion of Scores')
    plt.xlabel('Scores')
    plt.title('Distribution of scores obtained with a random-choice policy over 200 games')
    plt.legend()
    plt.show()

def get_model_distrib(a,b,c,d,e):
    fusion =[]
    for j in range(10):
        fusion.append(a[50+j])
        fusion.append(b[50+j])
        fusion.append(c[50+j])
        fusion.append(d[50+j])
        fusion.append(e[50+j])
    max =0.0
    for i in range(50):
        if(fusion[i]>max):
            max = fusion[i]
    distrib= [0 for i in range(int(max+1))]
    for i in range(50):
        distrib[ int(fusion[i]) ]+=1
    for i in range(int(max+1)):
        distrib[i]/=50
    return distrib


def plot_model_distrib():
    model_distrib = get_model_distrib(data_score1,data_score2,data_score3,data_score4,data_score5)
    scores_distrib = [i for i in range(len(model_distrib))]
    plt.barh(scores_distrib, model_distrib, align='center', alpha=0.5)
    
    means = [mean_trained_score for i in range(len(model_distrib))]
    means_noZero = [mean_trained_score_noZero for i in range(len(model_distrib))]
    
    plt.plot(model_distrib,means,c='r', label= "average score for the trained model")
    plt.plot(model_distrib,means_noZero,c='g', label= "average score for the trained model, discarding null scores")
    
    plt.ylabel('Scores')
    plt.xlabel('Proportion of Scores')
    plt.title('Distribution of scores obtained trained model over 50 games')
    plt.legend()
    plt.show()

plot_random_distrib()
#plot_model_distrib()