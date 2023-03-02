from waitress import serve
import main

serve(main.app, host='0.0.0.0',port=5000)
# serve(main.app, host='127.0.0.1',port=5000)