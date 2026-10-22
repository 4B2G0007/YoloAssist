"""Single source of truth for new recordings and actual spoken text."""
from .config import CLASSES, NEW_CLASSES
DIRECTIONS={'left':'左前方','center':'前方','right':'右前方'}

def additions():
    result={}
    for cls in sorted(NEW_CLASSES):
        name,zh=CLASSES[cls];name=name.replace(' ','_')
        for direction,word in DIRECTIONS.items():
            result[f'normal_{direction}_{name}.wav']=f'{word}有{zh}。'
        result[f'urgent_center_{name}.wav']=f'注意，前方有{zh}。'
    for cls in [0,1,2,3,5,7]:
        name,zh=CLASSES[cls];name=name.replace(' ','_')
        for side,word in [('left','左側'),('right','右側')]:
            result[f'entry_{side}_{name}.wav']=f'注意，{word}{zh}進入前方。'
    result.update({'quality_dark.wav':'畫面太暗，辨識可能受影響。',
                   'quality_unclear.wav':'畫面不清楚，請檢查攝影機。',
                   'quality_recovered.wav':'畫面品質已恢復。',
                   'summary_empty.wav':'目前沒有辨識到目標，請留意周圍環境。',
                   'summary_unavailable.wav':'目前影像資訊不足，暫時無法描述。'})
    return result

def event_clips(event):
    if 'clips' in event:return event['clips']
    if event.get('kind')=='camera_error':return ['camera_error.wav']
    return [event.get('clip',f"{event.get('level')}_{event.get('direction')}_{event.get('class_name')}.wav")]
