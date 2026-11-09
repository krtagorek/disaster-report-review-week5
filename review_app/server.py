"""Loopback-only HTTP UI/API. Standard library, no remote services or CDN."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .core import Archive, Store, Problem, LABELS, REASONS, ROOT


class AppServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,store,archive,model):
        self.store,self.archive,self.model=store,archive,model
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):
        pass  # Reports never enter request logs.

    def send(self,status,data,content_type='application/json',headers=None):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        for k,v in (headers or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(data)

    def dispatch(self,method):
        try:
            port=self.server.server_address[1]
            host=self.headers.get('Host','')
            if host not in (f'localhost:{port}',f'127.0.0.1:{port}'):
                raise Problem('Use this app through its local address.',403)
            if method!='GET':
                if self.headers.get('X-Review-App')!='1':raise Problem('Missing app request header.',403)
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://localhost:{port}',f'http://127.0.0.1:{port}'):
                    raise Problem('Requests must originate from this local app.',403)
            url=urlparse(self.path);path=url.path;query=parse_qs(url.query)
            state=query.get('state',[None])[0] or None
            if method=='GET':
                if path in ('/','/app.js','/style.css'):
                    filename={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
                    mime={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[filename]
                    return self.send(200,(Path(__file__).parent/'static'/filename).read_bytes(),mime)
                if path=='/api/health':return self.send(200,{'service':'disaster-report-review','ready':True,'model':self.server.model.metadata,'labels':LABELS,'reasons':REASONS})
                if path=='/api/archive':return self.send(200,self.server.archive.search(query.get('q',[''])[0]))
                if path.startswith('/api/archive/'):return self.send(200,self.server.archive.get(path.rsplit('/',1)[-1]))
                if path=='/api/reports':return self.send(200,self.server.store.list(state))
                if path.startswith('/api/reports/'):return self.send(200,self.server.store.get(path.rsplit('/',1)[-1]))
                if path=='/api/export':
                    fmt=query.get('format',['csv'])[0]
                    data,mime=self.server.store.export(fmt,state)
                    return self.send(200,data,mime,{'Content-Disposition':f'attachment; filename="disaster-reviews.{fmt}"'})
            else:
                try:length=int(self.headers.get('Content-Length','0'))
                except ValueError:raise Problem('Invalid request length.')
                if not 0<length<=150000:raise Problem('Request is empty or too large.',413)
                try:body=json.loads(self.rfile.read(length))
                except (ValueError,UnicodeError):raise Problem('Invalid JSON request.')
                if not isinstance(body,dict):raise Problem('Expected a JSON object.')
                if method=='POST' and path=='/api/classify':
                    text=body.get('text')
                    result=self.server.model.classify(text)
                    return self.send(200,self.server.store.classification(text,result))
                if method=='POST' and path=='/api/reports':
                    return self.send(201,self.server.store.create(body,self.server.archive))
                if method=='PUT' and path.startswith('/api/reports/'):
                    return self.send(200,self.server.store.update(path.rsplit('/',1)[-1],body))
            raise Problem('Not found.',404)
        except Problem as error:self.send(error.status,{'error':str(error)})
        except Exception as error:
            print(type(error).__name__+': '+str(error),flush=True)
            self.send(500,{'error':'The local app could not complete that request. Your saved reviews remain available.'})

    def do_GET(self):self.dispatch('GET')
    def do_POST(self):self.dispatch('POST')
    def do_PUT(self):self.dispatch('PUT')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8877)
    parser.add_argument('--database',type=Path,default=ROOT/'runtime/review-app.sqlite3')
    args=parser.parse_args()
    if not args.database.resolve().is_relative_to(ROOT/'runtime'):
        parser.error('Database must be inside this project runtime directory.')
    from .model import ServedModel
    print('Loading verified local merged model…',flush=True)
    model=ServedModel()
    server=AppServer(('127.0.0.1',args.port),Store(args.database),Archive(),model)
    print(f'Local review app: http://127.0.0.1:{server.server_address[1]}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();model.close()


if __name__=='__main__':main()
