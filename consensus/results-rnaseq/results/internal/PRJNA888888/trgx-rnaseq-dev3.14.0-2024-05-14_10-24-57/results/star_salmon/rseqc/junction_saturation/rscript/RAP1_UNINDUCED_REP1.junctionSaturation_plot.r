pdf('RAP1_UNINDUCED_REP1.junctionSaturation_plot.pdf')
x=c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100)
y=c(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
z=c(5,8,12,15,17,20,23,27,27,28,29,29,30,32,34,35,37,39,41,42)
w=c(3,6,10,13,15,18,21,25,25,26,27,27,28,30,32,33,35,37,39,40)
m=max(0,0,0)
n=min(0,0,0)
plot(x,z/1000,xlab='percent of total reads',ylab='Number of splicing junctions (x1000)',type='o',col='blue',ylim=c(n,m))
points(x,y/1000,type='o',col='red')
points(x,w/1000,type='o',col='green')
legend(5,0, legend=c("All junctions","known junctions", "novel junctions"),col=c("blue","red","green"),lwd=1,pch=1)
dev.off()