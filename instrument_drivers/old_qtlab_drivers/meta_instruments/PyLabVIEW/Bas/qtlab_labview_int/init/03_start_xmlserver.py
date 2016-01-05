from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

remote_ins_server=None
remote_py_ser=None

if objsh.start_glibtcp_client('localhost',port=12002, nretry=3, timeout=5):
    remote_ins_server=objsh.helper.find_object('qtlab_ngijs:instrument_server')
    #remote_py_ser=objsh.helper.find_object('qtlab_ngijs:python_server')
else:
    raise Exception

def do_command(commandstring):
    try: 
       result=str(remote_py_ser.cmd(commandstring))
    except:
        #TODO
        result='error in command evaluation'
    return result

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_function(do_command)
server.serve_forever()
