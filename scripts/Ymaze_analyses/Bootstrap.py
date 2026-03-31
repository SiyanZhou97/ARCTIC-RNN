import numpy as np

def one_sample_bootstrap(A,a=0,repeat=1000,p=0.05):
    A=A[~np.isnan(A)]
    n=A.shape[0]
    
    def test_statistic(A,a):
        var=np.sum(np.square(A-np.mean(A)))/(n-1)
        return (np.mean(A)-a)/np.sqrt(var/n)
    
    true_t_value=test_statistic(A,a)
    
    t_distribution=np.zeros(repeat)
    A_centered=A-np.mean(A)+a
    for i in range(repeat):
        sample=np.random.choice(A_centered,size=n,replace=True)
        t_distribution[i]=test_statistic(sample,a)
    return np.sum(t_distribution>=true_t_value)/repeat,np.sum(t_distribution<=true_t_value)/repeat

def bootstrap_ci_studendized(A,repeat=1000,p=2.5):
    """
    Use studentized bootstrap
    """
    A=A[~np.isnan(A)]
    n=A.shape[0]
     
    def test_statistic(A,a):
        var=np.sum(np.square(A-np.mean(A)))/(n-1)
        if var==0:
            var=1e-4
        return (np.mean(A)-a)/np.sqrt(var/n)
    
    meanA=np.mean(A)
    varA=np.sum(np.square(A-np.mean(A)))/(n-1)
    if varA==0:
        varA=1e-4
    seA=np.sqrt(varA/n)
    t_distribution=np.zeros(repeat)
    for i in range(repeat):
        sample=np.random.choice(A,size=n,replace=True)
        t_distribution[i]=test_statistic(sample,meanA)
    t25=np.percentile(t_distribution,p)
    t975=np.percentile(t_distribution,100-p)
    return meanA-t975*seA,meanA-t25*seA

def bootstrap_ci(A,repeat=1000,p=2.5):
    """
    Use Percentile bootstrap
    """
    A=A[~np.isnan(A)]
    n=A.shape[0]
     
    mean_distribution=np.zeros(repeat)
    for i in range(repeat):
        sample=np.random.choice(A,size=n,replace=True)
        mean_distribution[i]=np.mean(sample)
    t25=np.percentile(mean_distribution,p)
    t975=np.percentile(mean_distribution,100-p)
    return t25,t975

def two_sample_bootstrap(A,B,repeat=1000,p=0.05):
    A=A[~np.isnan(A)]
    B=B[~np.isnan(B)]
    n=A.shape[0]
    m=B.shape[0]
    
    def test_statistic(A,B):
        varA=np.sum(np.square(A-np.mean(A)))/(n-1)
        varB=np.sum(np.square(B-np.mean(B)))/(m-1)
        return (np.mean(A)-np.mean(B))/(np.sqrt(varA/n+varB/m))
    
    true_t_value=test_statistic(A,B)
    
    t_distribution=np.zeros(repeat)
    A_centered=A-np.mean(A)+np.mean(np.hstack((A,B)))
    B_centered=B-np.mean(B)+np.mean(np.hstack((A,B)))
    for i in range(repeat):
        sampleA=np.random.choice(A_centered,size=n,replace=True)
        sampleB=np.random.choice(B_centered,size=m,replace=True)
        t_distribution[i]=test_statistic(sampleA,sampleB)
    return np.sum(t_distribution>=true_t_value)/repeat,np.sum(t_distribution<=true_t_value)/repeat
