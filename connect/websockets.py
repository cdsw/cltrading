import websocket
import json

def on_message(ws, message):
    print(f"Received message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Closed connection: {close_status_code} {close_msg}")

def on_open(ws):
    print("Connection opened")
    # Subscribe to BTCUSDT trade stream
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [
            "btcusdt@kline_1m"  # Replace with your desired symbol and interval
        ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()