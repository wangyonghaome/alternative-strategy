
from reader import BarReader

with BarReader('/home/ywang/reader/bar.yaml') as client:
    bars = client.index_bars('000016')
    #bars = client.stock_bars('000001')
    print(bars)
