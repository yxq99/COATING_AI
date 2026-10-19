from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from coating_ai.eis_workflow import prepare_data, write_measurement_templates, MEASURE_COLUMNS
from coating_ai.config import load_project_config
from coating_ai.__main__ import build_parser


class EISTests(unittest.TestCase):
    def tables(self):
        forms=pd.DataFrame([['P1',.2,.5,'G1'],['N1',.7,1.2,'G2']],
                           columns=['sample_id','hybrid_wt_pct','ldh_mxene_ratio','validation_group'])
        rows=[]
        for day in [1,3,7,14,28,50]:
            rows.append(['P1','R1','P1S1',day,1e10,1,None,None,None])
            rows.append(['P1','R2','P1S2',day,None,None,None,None,None])
            rows.append(['N1','R1','N1S1',day,1e9 if day<=7 else None,1 if day<=7 else None,None,None,None])
        low=pd.DataFrame(rows,columns=MEASURE_COLUMNS)
        spectral=pd.DataFrame(columns=['sample_id','preparation_id','specimen_id','day','frequency_hz','z_real_ohm','z_imag_ohm'])
        params=pd.DataFrame([['target_day',28],['minimum_logZ',10],['hybrid_lower',.1],
                            ['hybrid_upper',1.5],['ratio_lower',.1],['ratio_upper',2],['risk_aversion',.25]],
                            columns=['parameter','value'])
        return {'配方':forms,'EIS低频':low,'EIS全谱':spectral,'新谱模板':spectral,'参数':params}

    def run_prepare(self,tables,folder):
        with patch('coating_ai.eis_workflow.pd.read_excel',
                   side_effect=lambda path,sheet_name:tables[sheet_name].copy()):
            return prepare_data(Path('unused.xlsx'),folder)

    def test_partial_future_labels_not_used_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);result=self.run_prepare(self.tables(),folder)
            self.assertEqual(result['design_rows'],1)
            self.assertEqual(result['pending_early_rows'],1)
            self.assertEqual(pd.read_csv(folder/'design_training.csv').sample_id.tolist(),['P1'])
            self.assertEqual(pd.read_csv(folder/'early_prediction_inputs.csv').sample_id.tolist(),['N1'])
            stats=pd.read_csv(folder/'repeat_statistics.csv')
            self.assertTrue((stats.n_preparations==1).all())
            self.assertTrue(stats.sd_logZ.isna().all())
            config=load_project_config(folder/'early_model.yaml')
            self.assertEqual(config.task,'prediction')
            self.assertEqual(config.integer_indices,[])

    def test_two_preparations_sd_and_mean_of_logs(self):
        tables=self.tables()
        low=tables['EIS低频']
        mask=low.sample_id.eq('P1') & low.preparation_id.eq('R2')
        low.loc[mask,'z_001_ohm']=1e8;low.loc[mask,'area_cm2']=1
        with tempfile.TemporaryDirectory() as temp:
            self.run_prepare(tables,Path(temp))
            row=pd.read_csv(Path(temp)/'repeat_statistics.csv').query("sample_id=='P1' and day==28").iloc[0]
            self.assertEqual(row.n_preparations,2)
            self.assertAlmostEqual(row.mean_logZ,9)
            self.assertAlmostEqual(row.sd_logZ,np.sqrt(2))

    def test_missing_area_is_not_assumed_or_imputed(self):
        tables=self.tables();tables['EIS低频'].loc[0,'area_cm2']=None
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'面积'):
                self.run_prepare(tables,Path(temp))

    def test_recommendation_templates_have_no_fake_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            write_measurement_templates(pd.DataFrame([['N1',.2,.5]],
                         columns=['sample_id','hybrid_wt_pct','ldh_mxene_ratio']),folder)
            low=pd.read_csv(folder/'measurements.csv')
            self.assertEqual(len(low),18)
            self.assertTrue(low.z_001_ohm.isna().all())
            self.assertTrue(low.area_cm2.isna().all())

    def test_parser_has_one_pipeline_without_demo_switch(self):
        parser=build_parser()
        args=parser.parse_args(['prepare-eis'])
        self.assertEqual(args.command,'prepare-eis')
        args=parser.parse_args(['predict','--run','model','--data','input.csv'])
        self.assertEqual(args.command,'predict')


if __name__=='__main__':unittest.main()
