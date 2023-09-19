from openpyxl import load_workbook
book = load_workbook('dataset.xlsx')
sheet = book.active
first_column = sheet['A']
with open('data.txt', 'w', encoding="utf-8") as outfile:
    for x in range(len(first_column)):
        outfile.write(str(first_column[x].value)+'\n')