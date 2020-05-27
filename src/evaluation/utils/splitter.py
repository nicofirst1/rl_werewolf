import csv



if __name__ == '__main__':

    file_name= '/Resources/CSVs/practical_ww_revenge.csv'

    with open(file_name,"r+") as f:
        rows = csv.reader(f, delimiter=',')
        rows=list(rows)

    headers= rows[0]

    odd_r=[headers]
    even_r=[headers]

    for r in rows[1:]:
        if int(r[0])%2==0:
            even_r.append(r)
        else:
            odd_r.append(r)

    base_name=file_name.split(".")[0]

    with open(f"{base_name}_odd.csv","w") as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(odd_r)

    with open(f"{base_name}_even.csv", "w") as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(even_r)