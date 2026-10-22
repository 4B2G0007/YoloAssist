import copy
import unittest
from assist.config import DEFAULT,validate
from assist.engine import Engine

def d(box=(400,200,600,700),cls=0,conf=.9):
    return {'class_id':cls,'confidence':conf,'xyxy':list(box)}

class RuleTests(unittest.TestCase):
    def setUp(self):
        self.c=copy.deepcopy(DEFAULT)
        self.c['alert_mode']='classic'
        self.e=Engine(self.c)
    def step(self,ds,t):
        return self.e.update(ds,1000,1000,t)
    def test_confirm_and_urgent_bypass(self):
        rows,_,ch,_=self.step([d()],0)
        self.assertEqual((rows[0]['state'],ch['level']),('pending','normal'))
        rows,_,ch,_=self.step([d()],.1)
        self.assertEqual((rows[0]['state'],ch['level'],rows[0]['track_id']),('near','urgent',1))
    def test_immediate(self):
        rows,_,ch,_=self.step([d((400,100,600,900))],0)
        self.assertEqual(rows[0]['reason'],'immediate_threshold')
        self.assertEqual(ch['level'],'urgent')
    def test_lateral(self):
        rows,_,ch,_=self.step([d((0,0,200,950))],0)
        self.assertFalse(rows[0]['in_roi'])
        self.assertEqual((ch['level'],ch['direction']),('normal','left'))
    def test_center_intrusion(self):
        rows,_,ch,_=self.step([d((0,0,600,950),2)],0)
        self.assertEqual(rows[0]['direction'],'left')
        self.assertEqual((ch['direction'],ch['level']),('center','urgent'))
    def test_cooldown(self):
        self.step([d()],0)
        self.step([d()],.1)
        _,ev,ch,_=self.step([d()],.2)
        self.assertIsNone(ch)
        self.assertEqual(ev[0]['reason'],'cooldown')
        self.step([d()],1.0)
        _,_,ch,_=self.step([d()],2.2)
        self.assertIsNotNone(ch)
    def test_missing_breaks_confirmation(self):
        self.step([d()],0)
        self.step([],.1)
        rows,_,_,_=self.step([d()],.2)
        self.assertEqual((rows[0]['consecutive_hits'],rows[0]['state']),(1,'pending'))
    def test_hysteresis(self):
        self.step([d()],0)
        self.step([d()],.1)
        rows,_,_,_=self.step([d((400,200,600,600))],.2)
        self.assertEqual(rows[0]['state'],'near')
        rows,_,_,_=self.step([d((400,200,600,550))],.3)
        self.assertEqual(rows[0]['state'],'detected')
    def test_expiry(self):
        self.step([d()],0)
        rows,_,_,tr=self.step([d()],3)
        self.assertNotEqual(rows[0]['track_id'],1)
        self.assertTrue(any(t['to']=='lost' for t in tr))
    def test_one_to_one(self):
        self.step([d()],0)
        rows,_,_,_=self.step([d(),d((410,200,610,700))],.1)
        self.assertEqual(len({r['track_id'] for r in rows}),2)
    def test_filter_reasons(self):
        rows,_,ch,_=self.step([d(conf=.3),d(cls=9)],0)
        self.assertEqual([r['reason'] for r in rows],['below_alert_confidence','non_target_class'])
        self.assertIsNone(ch)
    def test_priority(self):
        _,ev,ch,_=self.step([d((0,0,200,300),2),d((400,0,600,900))],0)
        self.assertEqual(ch['class_name'],'person')
        self.assertEqual(ev[1]['reason'],'lower_priority')
    def test_validation(self):
        for key,value in [('stride',0),('alert_conf',float('nan')),('confirm_hits',1.5)]:
            c=copy.deepcopy(DEFAULT);c[key]=value
            with self.assertRaises(ValueError):validate(c)
    def test_source_time_determinism(self):
        a,b=Engine(self.c),Engine(self.c)
        xs=[];ys=[]
        for t in [0,.1,.2,1,2.5,4]:
            xs.append(a.update([d()],1000,1000,t)[1])
            ys.append(b.update([d()],1000,1000,t)[1])
        self.assertEqual(xs,ys)
