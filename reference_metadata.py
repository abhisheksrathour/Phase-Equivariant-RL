from pathlib import Path
import urllib.request,urllib.parse,json,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[1]
DOIS=['10.1109/JSTQE.2010.2044751','10.1109/JLT.2008.927791','10.1109/JSAC.2020.3036950','10.1109/JSTQE.2022.3174268','10.1109/JLT.2024.3401419','10.1364/JOCN.483600','10.1109/JLT.2025.3550277','10.1109/JLT.2025.3555655','10.1016/j.optlastec.2025.112942',None,None,'10.2139/ssrn.7370318','10.1016/j.optcom.2025.132490','10.1364/OL.613094','10.1364/OE.589389','10.1364/OE.545577','10.1364/OL.599108','10.1109/JLT.2024.3386886','10.1109/LPT.2012.2187439','10.1364/OE.566763','10.1109/JLT.2026.3720192','10.1016/j.optcom.2025.132837','10.1109/JLT.2026.3653701','10.1109/JLT.2021.3108006','10.1109/50.622902']
def get(item):
 n,doi=item
 if not doi:return n,{}
 cache=ROOT/'references'/f'crossref_{n:02}.json'
 if cache.exists():return n,json.loads(cache.read_text())
 try:
  req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),headers={'User-Agent':'AcademicMetadataCheck/1.0'})
  m=json.load(urllib.request.urlopen(req,timeout=40))['message'];keep={k:m[k] for k in ['DOI','title','author','container-title','volume','issue','page','article-number','published','published-online','published-print','created','URL','type'] if k in m};cache.write_text(json.dumps(keep,ensure_ascii=False,indent=2));return n,keep
 except Exception as e:return n,{'error':str(e),'DOI':doi}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for n,m in sorted(pool.map(get,enumerate(DOIS,1))):print(n,m.get('DOI'),m.get('title'),m.get('published'),m.get('volume'),m.get('issue'),m.get('page'),m.get('error'))
