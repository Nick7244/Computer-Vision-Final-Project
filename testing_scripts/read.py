#!/bin/python3

import xlwt
from xlwt import Workbook

f = open("inception_metrics.txt")

wb = Workbook()
sheet1 = wb.add_sheet("Sheet 1")

loss, acc, val_loss, val_acc = [], [], [], []

for line in f:
    tokens = line.split()

    if len(tokens) == 0:
        continue

    if tokens[0] == "110/110":

        loss.append(tokens[7])
        acc.append(tokens[10])
        val_loss.append(tokens[13])
        val_acc.append(tokens[16])


for epoch in range(0, 25):
    sheet1.write(epoch, 0, epoch)
    sheet1.write(epoch, 1, loss[epoch])
    sheet1.write(epoch, 2, acc[epoch])
    sheet1.write(epoch, 3, val_loss[epoch])
    sheet1.write(epoch, 4, val_acc[epoch])
    
wb.save("inc_results.xls")
