"""Fresh on-demand summaries; no inferred free-space instructions or exact counts."""
from .config import CLASSES
from .voice_catalog import DIRECTIONS


def make_summary(engine, rows, now, valid, quality_ok=True):
    if now-engine.last_summary<engine.c['summary_cooldown_s']:
        e=engine.auxiliary(now,'summary','',clips=[])
        e.update(decision='suppressed',reason='summary_cooldown')
        return e
    engine.last_summary=now
    if not valid or not quality_ok:
        return engine.auxiliary(now,'summary','目前影像資訊不足，暫時無法描述。',clip='summary_unavailable.wav')
    objects=[r for r in rows if r.get('eligible')]
    objects.sort(key=lambda r:(r['state']=='near',r['in_roi'],r['height_ratio']),reverse=True)
    picked=[];seen=set()
    for row in objects:
        key=(row['class_name'],row['direction'])
        if key in seen:continue
        seen.add(key);picked.append(row)
        if len(picked)>=engine.c['summary_max_items']:break
    if not picked:
        return engine.auxiliary(now,'summary','目前沒有辨識到目標，請留意周圍環境。',clip='summary_empty.wav')
    text=''.join(f"{DIRECTIONS[r['direction']]}有{CLASSES[r['class_id']][1]}。" for r in picked)
    clips=[f"normal_{r['direction']}_{r['class_name'].replace(' ','_')}.wav" for r in picked]
    return engine.auxiliary(now,'summary',text,clips=clips,summary_items=[{'class_name':r['class_name'],'direction':r['direction']} for r in picked])
