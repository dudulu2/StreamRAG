import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from streamrag.attribution import prepare, validate, read_jsonl
from streamrag.three_layer import build, search, projection, load_reviewed, topical_query

W=ROOT/"corpus_sources/gemini_batches/nailv_20261016/windows.jsonl"
A=ROOT/"gemini_output/gemini_output/gemini_output/nailv_20261016/attribution/nailv_three_layer_attribution.jsonl"
P=ROOT/"data/attribution/nailv_20261016/semantic_patch.json"
R=ROOT/"data/attribution/nailv_20261016/revision_02"

class FakeEmbedding:
    mode="test"
    model="unit-test"
    contract={"metric":"cosine","dimensions":8,"version":1}
    def embed(self,texts):
        return [[(b+1)/256 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]
    def embed_query(self,text):return self.embed([text])[0]

class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in (W,A,P,R/"manifest.json")):
            raise unittest.SkipTest("Local reviewed corpus is intentionally not published")
        cls.units,cls.annotations,_,cls.audit,cls.report=prepare(W,A,P)
        cls.windows={w["id"]:w for w in read_jsonl(W)}
    def unit(self,suffix):
        return next(u for u in self.units if u["id"].endswith(suffix))
    def test_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"windows.jsonl";p.write_bytes(W.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"Source changed"):prepare(p,A,P)
    def test_exact_original_quotes_and_coverage(self):
        self.assertEqual(41,self.report["windows"])
        self.assertEqual(160,len(self.audit))
        self.assertEqual(0,self.report["conflicts"])
        self.assertEqual(450,len(self.units))
        self.assertTrue(all(not u["speaker_verified"] for u in self.units))
    def test_childhood_essay_is_not_viewer(self):
        f=self.unit("memories_part2__00018__fact_02")
        v=self.unit("memories_part2__00018__sub_01")
        self.assertIn("甚至作文还拿过高分",f["evidence"][0]["text"])
        self.assertFalse(any("我小时候写作文" in s["text"] for s in v["evidence"]))
    def test_scent_questions_removed(self):
        f=self.unit("daytime_part2__00003__fact_01")
        text="\n".join(s["text"] for s in f["evidence"])
        for q in ("喜欢香料的味道吗","啥牌子","雪松是啥味"):
            self.assertNotIn(q,text)
        self.assertIn("雪松",text)
    def test_indirect_injury_is_not_fabricated_viewer_quote(self):
        u=self.unit("emotion_part2__00001__inter_01")
        self.assertEqual([],u["trigger_evidence"])
        self.assertEqual("context_inferred",u["trigger_kind"])
        self.assertTrue(u["evidence"])
    def test_cross_window_trigger_has_exact_evidence(self):
        u=self.unit("emotion_part2__00014__inter_01")
        self.assertEqual([],u["trigger_evidence"])
        self.assertEqual("context_inferred",u["trigger_kind"])
        self.assertTrue(u["trigger_context"])
        self.assertTrue(all(s["window_id"].endswith("00013") for s in u["trigger_context"]))
    def test_mixed_viewer_span_is_rejected(self):
        units=copy.deepcopy(self.units)
        u=next(u for u in units if u["id"].endswith("memories_part2__00018__sub_01"))
        u["evidence"]=copy.deepcopy(u["context_evidence"])
        u["evidence"][0]["author"]="viewer"
        with self.assertRaisesRegex(ValueError,"Attribution conflict"):validate(units,self.windows,self.annotations)
    def test_quote_edit_and_layer_mismatch_are_rejected(self):
        units=copy.deepcopy(self.units);u=next(u for u in units if u["kind"]=="quote")
        u["evidence"][0]["text"]+="改写"
        with self.assertRaisesRegex(ValueError,"Evidence text"):validate(units,self.windows,self.annotations)
        units=copy.deepcopy(self.units);units[0]["layer"]=2
        with self.assertRaisesRegex(ValueError,"kind/layer"):validate(units,self.windows,self.annotations)
    def test_projection_does_not_leak_mixed_context(self):
        u=self.unit("memories_part2__00018__fact_02")
        self.assertNotIn("context_evidence",projection(u))
        self.assertNotIn("raw_text",projection(u))
    def test_uncertainty_and_hypothetical_status(self):
        self.assertEqual("context_only",self.unit("emotion_part2__00001__sub_02")["review_status"])
        self.assertEqual("hypothetical",self.unit("memories_part2__00010__fact_01")["epistemic_status"])
        self.assertEqual("hearsay",self.unit("emotion_part2__00001__fact_02")["epistemic_status"])
    def test_saved_revision_is_immutable_and_valid(self):
        units,_=load_reviewed(R,W)
        self.assertEqual(self.units,units)

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in (W,R/"manifest.json",R/"records.jsonl",R/"author_spans.jsonl")):
            raise unittest.SkipTest("Local reviewed corpus is intentionally not published")
        cls.temp=tempfile.TemporaryDirectory();cls.index=Path(cls.temp.name)/"index"
        cls.embed=FakeEmbedding()
        build(R,W,cls.index,cls.embed)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_scope_and_epistemic_filter_before_limit(self):
        for scope,kinds in [("facts",{"fact"}),("style",{"quote","interaction"}),("attribution",{"fact","viewer"})]:
            rows=search(self.index,"观众相亲经历",self.embed,scope=scope,top_k=2)
            self.assertEqual(2,len(rows));self.assertTrue(all(r["kind"] in kinds for r in rows))
        rows=search(self.index,"断骨风险",self.embed,scope="facts",epistemic="hearsay")
        self.assertEqual(1,len(rows));self.assertEqual("hearsay",rows[0]["epistemic_status"])
    def test_threshold_applies_to_lexical_candidates(self):
        rows=search(self.index,"雪松香味",self.embed,min_similarity=1.0)
        self.assertEqual([],rows)
    def test_subject_name_is_not_a_topic(self):
        manifest={"speaker_ids":["nailv"],"speaker_aliases":{"nailv":["明前奶绿"]}}
        self.assertEqual("喜欢什么香味？",topical_query("明前奶绿喜欢什么香味？",manifest))
        self.assertEqual("明前奶绿",topical_query("明前奶绿",manifest))
        manifest["speaker_ids"].append("other")
        self.assertEqual("明前奶绿喜欢什么香味？",topical_query("明前奶绿喜欢什么香味？",manifest))
    def test_contract_mismatch_rejected(self):
        other=FakeEmbedding();other.contract=dict(other.contract,version=2)
        with self.assertRaisesRegex(ValueError,"contract mismatch"):search(self.index,"香味",other)
    def test_existing_index_not_overwritten(self):
        with self.assertRaisesRegex(ValueError,"Index exists"):build(R,W,self.index,self.embed)

if __name__=="__main__":unittest.main()
