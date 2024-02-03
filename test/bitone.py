
#Explanation is available at the following URLs (5 videos):
#https://youtu.be/e0IAvMUoXMQ
#https://youtu.be/Yp2u_qZHaMc
#https://youtu.be/3vcGa9ojDeM
#https://youtu.be/OlBUqXVJH24
#https://youtu.be/jfTequbukXY

import datetime as dt
from binance.client import Client
import webbrowser
import time
import numpy
import random
import tkinter
import requests
from datetime import datetime
import pandas as pd


#Copyright by Bitone Great    www.bitonegreat.com

root = tkinter.Tk()
root.title('Binance Futures MACD Program www.bitonegreat.com')
root.geometry("700x1000")

input1 = tkinter.Label(root, text="API Key")
input1.pack()

e1 = tkinter.Entry(root, show="*")
e1.pack()

input2 = tkinter.Label(root, text="API Secret")
input2.pack()

e2 = tkinter.Entry(root, show="*")
e2.pack()

input3 = tkinter.Label(root, text="Moving Average Fast (ex, 12)")
input3.pack()

e3 = tkinter.Entry(root)
e3.pack()

input4 = tkinter.Label(root, text="Moving Average Fast (ex, 26)")
input4.pack()

e4 = tkinter.Entry(root)
e4.pack()

input5 = tkinter.Label(root, text="Signal Line (ex, 9)")
input5.pack()

e5 = tkinter.Entry(root)
e5.pack()

input6 = tkinter.Label(root, text="Symbol (ex, BTCUSDT)")
input6.pack()

e6 = tkinter.Entry(root)
e6.pack()

input7 = tkinter.Label(root, text="Leverage (ex, 2)")
input7.pack()

e7 = tkinter.Entry(root)
e7.pack()

input8 = tkinter.Label(root, text="Time Interval (ex, 5)")
input8.pack()

e8 = tkinter.Entry(root)
e8.pack()

input9 = tkinter.Label(root, text="Order Amount in USDT (ex, 100)(Leave empty if not used)")
input9.pack()

e9 = tkinter.Entry(root)
e9.pack()

input10 = tkinter.Label(root, text="Order Amount in % (ex, 10)(Leave empty if not used)")
input10.pack()

e10 = tkinter.Entry(root)
e10.pack()

input11 = tkinter.Label(root, text="Loss cut rate % (ex, 10)")
input11.pack()

e11 = tkinter.Entry(root)
e11.pack()

input12 = tkinter.Label(root, text="Order Entry Value (ex, 5)")
input12.pack()

e12 = tkinter.Entry(root)
e12.pack()

input13 = tkinter.Label(root, text="Profit Take % (ex, 10)")
input13.pack()

e13 = tkinter.Entry(root)
e13.pack()

input14 = tkinter.Label(root, text="Profit Take 1 in % (ex, 10)")
input14.pack()

e14 = tkinter.Entry(root)
e14.pack()

input15 = tkinter.Label(root, text="Profit Take Stop 1 in % (ex, 10)")
input15.pack()

e15 = tkinter.Entry(root)
e15.pack()

input16 = tkinter.Label(root, text="Profit Take 2 in % (ex, 20)")
input16.pack()

e16 = tkinter.Entry(root)
e16.pack()

input17 = tkinter.Label(root, text="Profit Take Stop 2 in % (ex, 20)")
input17.pack()

e17 = tkinter.Entry(root)
e17.pack()

orderlist = []

now = datetime.now()

ptest1=0

#Copyright by Bitone Great    www.bitonegreat.com

#Explanation is available at the following URLs (5 videos):
#https://youtu.be/e0IAvMUoXMQ
#https://youtu.be/Yp2u_qZHaMc
#https://youtu.be/3vcGa9ojDeM
#https://youtu.be/OlBUqXVJH24
#https://youtu.be/jfTequbukXY


def tick():
    try:
        if not doTick:
            return
        global ptest1
        global execute
        global roelist
        api_key = str(e1.get())
        api_secret = str(e2.get())
        short = int(e3.get())
        long = int(e4.get())
        signal = int(e5.get())
        symbol = str(e6.get())
        leverage = int(e7.get())
        timeinterval = str(e8.get())
        buyd = float(e12.get())



        client = Client(api_key=api_key, api_secret=api_secret, testnet=False)

        try:
            absorderamount = float(e9.get())
        except:
            absorderamount = 0
        try:
            reorderamount = float(e10.get()) / 100
        except:
            reorderamount = 1

        if reorderamount >= leverage:
            leverage = round(leverage + (reorderamount - leverage), 1) + 1
            print(str(leverage)+'x'+' <-New Leverage')

        try:
            losscut = float(e11.get()) / 100
        except:
            losscut = 1

        try:
            profitcut = float(e13.get()) / 100
        except:
            profitcut = 1

        profittarget1=float(e14.get()) / 100
        profitcut1=float(e15.get()) / 100
        profittarget2=float(e16.get()) / 100
        profitcut2=float(e17.get()) / 100

        insert = 0
        insert2 = 0
        insert3 = 0

        now = datetime.now()
        current_time = now.strftime("(%H : %M : %S )")
        print(current_time)

        info = client.futures_exchange_info()

        for item in info['symbols']:
            if item['symbol'] == symbol:
                symbols_n_precision = item['quantityPrecision']
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except:
            pass

#Copyright by Bitone Great    www.bitonegreat.com

        try:
            client.futures_change_margin_type(symbol=symbol, marginType='CROSSED')
        except:
            pass
        time.sleep(0.5)

        url = 'https://fapi.binance.com/fapi/v1/klines?symbol=' + symbol + '&interval=' + str(timeinterval) + 'm'
        data = requests.get(url).json()

        D = pd.DataFrame(data)
        D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'is_best_match']
        D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
        D['symbol'] = symbol
        D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
               'taker_quote_vol']]

        df = D.set_index("open_date_time")

        marketprice = 'https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=' + symbol
        res = requests.get(marketprice)
        data = res.json()
        price = float(data['lastPrice'])

        df['open'] = df['open'].astype(float)

        df2 = df['open'].to_numpy()

        df2 = numpy.append(df2, [price])

        df = pd.DataFrame(df2, columns=['open'])

        exp1 = df['open'].ewm(span=short, adjust=False).mean()
        exp2 = df['open'].ewm(span=long, adjust=False).mean()
        macd = exp1 - exp2

        exp3 = macd.ewm(span=signal, adjust=False).mean()

        test1 = exp3.iloc[-1] - macd.iloc[-1]

        for i in range(2, len(df)):
            test2 = exp3.iloc[-i] - macd.iloc[-i]

            if test1>0 and test2<0:
               break
            if test1 < 0 and test2 > 0:
               break

#Copyright by Bitone Great    www.bitonegreat.com

        test3 = exp3.iloc[-2] - macd.iloc[-2]

        call = 'N/A'
        call1= 'N/A'

        if test1 < 0 and test2 > 0 and abs(test1) >= buyd:
            if test3/test1>0 and abs(test3) < buyd:
                call1 = 'Goldencross for entry'
            if test3/test1<0:
                call1 = 'Goldencross for entry'

        if test1 > 0 and test2 < 0 and abs(test1) >= buyd:
            if test3/test1>0 and abs(test3) < buyd:
                call1 = 'Deadcross for entry'
            if test3/test1<0:
                call1 = 'Deadcross for entry'

        if test1 < 0 and test2 > 0:
            call = 'Goldencross'

        if test1 > 0 and test2 < 0:
            call = 'Deadcross'

        print(call)
        print(call1)
        try:
            balance = client.futures_account_balance()
            time.sleep(0.25)
            account = client.futures_account()
        except Exception as e:
            print(e.message)
            pass

        usdtbalance = 0

        for b1 in balance:
            if b1['asset'] == 'USDT':
                usdtbalance = float(b1['withdrawAvailable'])

        for b1 in account['assets']:
            if b1['asset'] == 'USDT':
                initialmargin = float(b1['initialMargin'])
                unrealizedprofit = float(b1['unrealizedProfit'])

        try:
            roe = unrealizedprofit / initialmargin
        except:
            roe = 0

        if ptest1 == 0:
           roelist=[]

        if ptest1 != 0:
            print('1')
            roelist.append(roe)
            print(roelist)
            roe2=max(roelist)*profitcut1
            roe3=max(roelist) * profitcut2
            roelist=[max(roelist)]
            print(roe2)
            print(roe3)

        if ptest1!=3 and roe>=profittarget1:
           ptest1 = 1

        if ptest1==1 and roe<=roe2:
           ptest1 = 2

        if roe>=profittarget2:
           ptest1 = 3

        if ptest1==3 and roe<=roe3:
           ptest1 = 4


        try:
            print(roe2)
        except:
            pass
        try:
            print(roe3)
        except:
            pass
        if usdtbalance < absorderamount:
            leverage = round(absorderamount / usdtbalance + 0.5, 1)
            try:
                client.futures_change_leverage(symbol=symbol, leverage=leverage)
                print(str(leverage) +'x'+' <-New Leverage')
            except:
                pass

        symbolbalance = 0
        for c1 in account['positions']:
            if c1['symbol'] == symbol:
                symbolbalance = float(c1['positionAmt'])
                entryprice = float(c1['entryPrice'])

        orderbook = client.futures_order_book(symbol=symbol, limit=5)
        bid = float(orderbook['bids'][0][0])
        ask = float(orderbook['asks'][0][0])

        trade_size_in_dollars = usdtbalance * reorderamount

        if absorderamount != 0:
            trade_size_in_dollars = absorderamount

        order_amount = trade_size_in_dollars / price
        order_amount_buy = trade_size_in_dollars / ask
        order_amount_sell = trade_size_in_dollars / bid


        precision = symbols_n_precision

        if symbolbalance < 0:
            if roe < (losscut*-1):
                try:
                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=abs(symbolbalance))
                    print(buyorder)
                    print('Stop loss buy')
                    insert = current_time + symbol + ' Stop loss buy'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' Stop loss buy problem'
                    pass

            if roe > profitcut:
                try:
                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=abs(symbolbalance))
                    print(buyorder)
                    print('Profit take buy')
                    insert = current_time + symbol + ' Profit take buy'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' Profit take buy problem'
                    pass

            if ptest1==2:
                try:
                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=abs(symbolbalance))
                    print(buyorder)
                    print('Profit take buy (Profit Take Stop 1)')
                    insert = current_time + symbol + ' Profit take buy (Profit Take Stop 1)'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' Profit take buy problem'
                    pass

            if ptest1==4:
                try:
                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=abs(symbolbalance))
                    print(buyorder)
                    print('Profit take buy (Profit Take Stop 2)')
                    insert = current_time + symbol + ' Profit take buy (Profit Take Stop 2)'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + 'Profit take buy problem'
                    pass


        if symbolbalance > 0:
            if roe < (losscut*-1):
                try:
                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET',
                                                            quantity=abs(symbolbalance))
                    print(sellorder)
                    print('Loss cut sell')
                    insert = current_time + symbol + ' Loss cut sell'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' Loss cut sell problem'
                    pass
                
            if roe > profitcut:
                try:
                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET',
                                                            quantity=abs(symbolbalance))
                    print(sellorder)
                    print('Profit take sell')
                    insert = current_time + symbol + ' Profit take sell'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' Profit take sell problem'
                    pass

            if ptest1==2:
                try:
                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET',
                                                            quantity=abs(symbolbalance))
                    print(sellorder)
                    print('Profit take sell')
                    insert = current_time + symbol + ' Profit take sell (Profit Take Stop 1)'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + 'Profit take sell problem'
                    pass

            if ptest1==4:
                try:
                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET',
                                                            quantity=abs(symbolbalance))
                    print(sellorder)
                    print('Profit take sell')
                    insert = current_time + symbol + ' Profit take sell (Profit Take Stop 2)'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + 'Profit take sell problem'
                    pass
                
#Copyright by Bitone Great    www.bitonegreat.com

        if call == 'Goldencross':
            if symbolbalance < 0:

                try:
                    buyorderq = "{:0.0{}f}".format(abs(symbolbalance), precision)

                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=buyorderq)
                    print(buyorder)
                    print('Goldencross buy')
                    insert = current_time + symbol + ' Goldencross buy'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' buy problem'
                    pass

        if call1 == 'Goldencross for entry':
            if symbolbalance == 0:
                try:
                    buyorderq = "{:0.0{}f}".format(order_amount_buy, precision)
                    buyorder = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=buyorderq)
                    print(buyorder)
                    print('Goldencross buy')
                    insert = current_time + symbol + ' buy order'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' buy order problem'
                    pass

        if call == 'Deadcross':
            if symbolbalance > 0:
                try:
                    sellorderq = "{:0.0{}f}".format(abs(symbolbalance), precision)

                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=sellorderq)
                    print(sellorder)
                    print('Deadcross sell')
                    insert = current_time + symbol + ' sell order'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' sell order problem'
                    pass

        if call1 == 'Deadcross for entry':
            if symbolbalance == 0:
                try:
                    sellorderq = "{:0.0{}f}".format(order_amount_sell, precision)
                    sellorder = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=sellorderq)
                    print(sellorder)
                    print('Deadcross sell')
                    insert = current_time + symbol + ' sell order'
                    ptest1 = 0
                except Exception as e:
                    print(e.message)
                    insert2 = current_time + symbol + ' sell order problem'
                    pass
                
#Copyright by Bitone Great    www.bitonegreat.com

#Explanation is available at the following URLs (5 videos):
#https://youtu.be/e0IAvMUoXMQ
#https://youtu.be/Yp2u_qZHaMc
#https://youtu.be/3vcGa9ojDeM
#https://youtu.be/OlBUqXVJH24
#https://youtu.be/jfTequbukXY


        text1 = 'Moving Average Fast: ' + str(round(exp1.iloc[-1], 4))
        text2 = 'Moving Average Slow: ' + str(round(exp2.iloc[-1], 4))
        text3 = 'DIF: ' + str(round(macd.iloc[-2], 4))
        text4 = 'DEM: ' + str(round(exp3.iloc[-2], 4))
        text5 = 'Last price: ' + str(price) + ' USDT'
        text6 = 'Deadcross or Goldencross: ' + str(call)

        timeLabel.config(text=text1)
        timeLabel2.config(text=text2)
        timeLabel3.config(text=text3)
        timeLabel4.config(text=text4)
        timeLabel5.config(text=text5)
        timeLabel6.config(text=text6)

        if insert != 0:
            listbox.insert(0, insert)

        if insert2 != 0:
            listbox.insert(0, insert2)

        if insert3 != 0:
            listbox.insert(0, insert3)
        time.sleep(0.5)

    except Exception as e:
        print(e.message)
        time.sleep(1)
        pass

    if ptest1==1:
       print('Profit Take Target 1 Touched')

    if ptest1==3:
       print('Profit Take Target 2 Touched')

    root.after(1000, tick)
    
#Copyright by Bitone Great    www.bitonegreat.com

def stop():
    global doTick
    doTick = False


def start():
    global doTick
    doTick = True
    # Perhaps reset `sec` too?
    tick()

def callback(event):
    webbrowser.open_new('https://www.bitonegreat.com')

def callback2(event):
    webbrowser.open_new('https://www.youtube.com/watch?v=e0IAvMUoXMQ&t=520s')

timeLabel = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel.pack()

timeLabel2 = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel2.pack()

timeLabel3 = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel3.pack()

timeLabel4 = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel4.pack()

timeLabel5 = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel5.pack()

timeLabel6 = tkinter.Label(root, fg='green', font=('Helvetica', 10))
timeLabel6.pack()

startButton = tkinter.Button(root, text='Start', command=start)
startButton.pack()

stopButton = tkinter.Button(root, text='Stop', command=stop)
stopButton.pack()

lb2 = tkinter.Label(root, text=r"", fg="blue", cursor="hand2")
lb2.pack()

lb3 = tkinter.Label(root, text=r"For instruction, please click here", fg="blue", cursor="hand2")
lb3.pack()
lb3.bind("<Button-1>", callback2)

lbl = tkinter.Label(root, text=r"Copyright by Bitone Great", fg="blue", cursor="hand2")
lbl.pack()
lbl.bind("<Button-1>", callback)

scrollbar = tkinter.Scrollbar(root)
scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

listbox = tkinter.Listbox(root, yscrollcommand=scrollbar.set, width=700, height=100)
listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH)

root.mainloop()


#Copyright by Bitone Great    www.bitonegreat.com

#Explanation is available at the following URLs (5 videos):
#https://youtu.be/e0IAvMUoXMQ
#https://youtu.be/Yp2u_qZHaMc
#https://youtu.be/3vcGa9ojDeM
#https://youtu.be/OlBUqXVJH24
#https://youtu.be/jfTequbukXY