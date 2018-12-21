clc;
clear all;

block = [1 2 4 8 10 20];
time = [0.239584 0.090112 0.039872 0.027680 0.028640 0.024640];

figure (1)
plot(block,time,'k-*','MarkerSize',4);
title('Kernel time vs BLOCK\_SIZE');
axis([0 25 0 0.3])
xlabel('BLOCK\_SIZE');
ylabel('time(ms)');
legend('Kernel Execution Time');