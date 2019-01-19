# -*- coding: utf-8 -*-
import xlrd

def readExcelByName(fileName, sheetName):
	table = None
	errMsg = None
	try:
		data = xlrd.open_workbook(fileName)
		table = data.sheet_by_name(sheetName)
	except Exception as e:
		errMsg = e
	return table, errMsg

def getColumnIndex(table, colName):
	colIndex = None
	for i in range(table.ncols):
		if (table.cell_value(1,i) == colName):
			colIndex = i
			break
	return colIndex

def getColValues(table, colIndex):
	colValues = []
	for i in range(3, table.nrows):
		colValues.append(str(table.cell_value(i, colIndex)))
	return colValues

def save(fileName, contents):
	fh = open(fileName, 'w', encoding='utf-8')
	fh.write(contents)
	fh.close()

table, errMsg = readExcelByName('./comments.xlsx', 'Sheet4')
colIndex = getColumnIndex(table, u'评论内容')
colValues = getColValues(table, colIndex)
comments = '\n'.join(colValues)
save('comments.txt', comments)