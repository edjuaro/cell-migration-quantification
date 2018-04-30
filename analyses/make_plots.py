import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


names = np.chararray( 20,unicode=True, itemsize=15)
percent = np.zeros(20)

f = open('output_c3.txt')
i = 0
for line in f.readlines():
    # print(line)
    templine = line.split('\t')
    # print(templine)
    temp = templine[0]
    temp = temp.strip('.tif').split('_')
    names[i] = temp[1]
    percent[i] = float(templine[3])
    print(names[i],percent[i])
    i+=1

H10 = percent[0:4]
ctrl_H10 = percent[4:8]
Taxol = percent[8:12]
stopper = percent[12:16]
untreated = percent[16:20]

x = np.array([1,2,3,4,5])
width = 0.8
means = [np.average(untreated),np.average(stopper),np.average(Taxol),np.average(H10),np.average(ctrl_H10)]
var =  [np.var(untreated),np.var(stopper),np.var(Taxol),np.var(H10),np.var(ctrl_H10)]
sem = [stats.sem(untreated),stats.sem(stopper),stats.sem(Taxol),stats.sem(H10),stats.sem(ctrl_H10)]

for i in [0,1,2,3]:
    plt.errorbar(x=1+width*0.125+width*(i*0.25),y=untreated[i],fmt='wo')
    plt.errorbar(x=2+width*0.125+width*(i*0.25),y=stopper[i],fmt='wo')
    plt.errorbar(x=3+width*0.125+width*(i*0.25),y=Taxol[i],fmt='wo')
    plt.errorbar(x=4+width*0.125+width*(i*0.25),y=H10[i],fmt='wo')
    plt.errorbar(x=5+width*0.125+width*(i*0.25),y=ctrl_H10[i],fmt='wo')

# plt.bar(left=x,height=means,yerr=sem,width=width)
plt.bar(left=x,height=means,width=width)
ax = plt.gca()
ax.set_xticks(x + width/2)
ax.set_xticklabels(('Untreated', 'Stopper', 'Taxol', 'AGR2-Ab','IgG (Ctrl-Ab)'))
plt.xlim(xmin=width,xmax=6)

vals = ax.get_yticks()
ax.set_yticklabels(['{:4.0f}%'.format(x) for x in vals])


plt.ylabel('Migration region covered by cells')
plt.xlabel('Experimental condition')
plt.savefig('comparison.png')

SE_ctrl = np.sqrt(var[0]/4+var[4]/4)
DF_ctrl = np.round((var[0]/4+var[4]/4)**2 / ( (var[0]/4)**2/3 + (var[4]/4)**2/3 ))
t_ctrl = (means[4]-means[0])/SE_ctrl
p_ctrl = stats.t.cdf(x=t_ctrl,df=DF_ctrl)
print(SE_ctrl,DF_ctrl,t_ctrl,p_ctrl)

SE_h10 = np.sqrt(var[0]/4+var[3]/4)
DF_h10 = np.round((var[0]/4+var[3]/4)**2 / ( (var[0]/4)**2/3 + (var[3]/4)**2/3 ))
t_h10 = (means[3]-means[0])/SE_h10
p_h10 = stats.t.cdf(x=t_h10,df=DF_h10)
print(SE_h10,DF_h10,t_h10,p_h10)

SE_tax = np.sqrt(var[0]/4+var[2]/4)
DF_tax = np.round((var[0]/4+var[2]/4)**2 / ( (var[0]/4)**2/3 + (var[2]/4)**2/3 ))
t_tax = (means[2]-means[0])/SE_tax
p_tax = stats.t.cdf(x=t_tax,df=DF_tax)
print(SE_tax,DF_tax,t_tax,p_tax)

print(means)
print(sem)

