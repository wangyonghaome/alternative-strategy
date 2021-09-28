from reader.reader import BarReader
with BarReader('/home/ywang/proj_cross_grid/reader/bar.yaml') as client:
    bars = client.stock_bars('000001')
print(bars)