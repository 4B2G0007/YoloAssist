"""Audit existing Clipchamp clips and V1.1 additions, without speech libraries."""
import argparse,json,sys,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assist.voice_catalog import additions

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir',default=str(root/'audio'))
    args=parser.parse_args();folder=Path(args.audio_dir)
    original=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    expected={**{name:item['text'] for name,item in original.items()},**additions()}
    missing=[];bad=[];valid=[]
    for name in expected:
        path=folder/name
        if not path.is_file():missing.append(name);continue
        try:
            with wave.open(str(path)) as f:
                if f.getcomptype()!='NONE' or f.getnframes()==0 or f.getsampwidth()!=2 or f.getnchannels()!=1:
                    raise ValueError('Expected nonempty mono 16-bit PCM WAV')
            valid.append(name)
        except Exception as exc:bad.append({'file':name,'error':str(exc)})
    print(json.dumps({'expected':len(expected),'valid':len(valid),'missing':missing,'invalid':bad},ensure_ascii=False,indent=2))
    return 1 if missing or bad else 0
if __name__=='__main__':sys.exit(main())
