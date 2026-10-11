import hashlib
import json
import pytest

from artifacts.calibration_release import require_calibration_release, require_calibration_assets


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path):
    folder=tmp_path/'calibration/synthetic'
    folder.mkdir(parents=True)
    source=tmp_path/'code/frozen.py';source.parent.mkdir();source.write_text('# frozen')
    audit_script=folder/'audit_completed_outputs.py';audit_script.write_text('# audit')
    (folder/'design.json').write_text(json.dumps({'code_sha':{str(source):digest(source)}}))
    (folder/'run_full.json').write_text(json.dumps({'expected_scores':480,'recorded_scores':480}))
    (folder/'summary_full.csv').write_text('completed\nTrue\n')
    (folder/'aggregate_full.csv').write_text('completed\n480\n')
    audit={'schema_version':1,'status':'PASS','errors':[],
           'expected_scores':480,'recorded_scores':480,'completed_scores':477,'failed_scores':3,
           'independently_rescored':477,'generated_datasets_audited':240,
           'allocated_unique_stage_seeds':1200,'generation_k_mismatches':0,
           'generation_initial_state_mismatches':0,'reference_k_mismatches_reported':0,
           'audit_script_sha':digest(audit_script),
           'artifact_hashes':{f'calibration/synthetic/{name}':digest(folder/name)
             for name in ('design.json','run_full.json','summary_full.csv','aggregate_full.csv')}}
    (folder/'final_output_audit.json').write_text(json.dumps(audit))
    return tmp_path,audit


def save_audit(root,audit):
    (root/'calibration/synthetic/final_output_audit.json').write_text(json.dumps(audit))


def test_release_accepts_audited_missing_arms_and_records_release_id(release):
    root,audit=release
    result=require_calibration_release(root)
    assert result['completed_scores']==477 and result['failed_scores']==3
    assert result['release_id']==digest(root/'calibration/synthetic/final_output_audit.json')


@pytest.mark.parametrize('field,value',[
    ('status','FAIL'),('errors',['mismatch']),('expected_scores',479),
    ('completed_scores',478),('failed_scores',-1),('independently_rescored',476),
    ('generated_datasets_audited',239),('allocated_unique_stage_seeds',1199),
    ('generation_k_mismatches',1),('generation_initial_state_mismatches',1),
    ('reference_k_mismatches_reported',1),('schema_version',0)])
def test_release_rejects_failed_or_incomplete_audit(release,field,value):
    root,audit=release;audit[field]=value;save_audit(root,audit)
    with pytest.raises(RuntimeError):require_calibration_release(root)


@pytest.mark.parametrize('relative',[
    'calibration/synthetic/aggregate_full.csv','calibration/synthetic/summary_full.csv',
    'calibration/synthetic/run_full.json','calibration/synthetic/design.json',
    'calibration/synthetic/audit_completed_outputs.py','code/frozen.py'])
def test_release_rejects_stale_files_or_changed_sources(release,relative):
    root,_=release
    with (root/relative).open('a') as out:out.write('\nchanged')
    with pytest.raises(RuntimeError):require_calibration_release(root)


def test_release_requires_audit_file(release):
    root,_=release;(root/'calibration/synthetic/final_output_audit.json').unlink()
    with pytest.raises(RuntimeError):require_calibration_release(root)


@pytest.fixture
def assembly(release):
    root,audit=release;state=require_calibration_release(root)
    folder=root/'figures/assembled';folder.mkdir(parents=True)
    paths=[folder/f'Figure_S5.{ext}' for ext in ('png','pdf','tiff')]
    paths += [folder/f'Figure_S5_part{i}.png' for i in (1,2)]
    for path in paths:path.write_bytes(b'figure')
    manifest={'release_id':state['release_id'],'assets':{p.relative_to(root).as_posix():digest(p) for p in paths}}
    (root/'logs').mkdir()
    target=root/'logs/calibration_assembly_release.json';target.write_text(json.dumps(manifest))
    return root,state,target,manifest


def test_assembly_bound_to_current_audit(assembly):
    root,state,_,_=assembly
    assert len(require_calibration_assets(root,state,assembled=True)['assets'])==5


@pytest.mark.parametrize('problem',['wrong_release','missing_asset','changed_image'])
def test_assembly_rejects_stale_or_incomplete_figures(assembly,problem):
    root,state,target,manifest=assembly
    if problem=='wrong_release':manifest['release_id']='stale'
    elif problem=='missing_asset':manifest['assets'].pop('figures/assembled/Figure_S5_part2.png')
    else:(root/'figures/assembled/Figure_S5.png').write_bytes(b'changed')
    target.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):require_calibration_assets(root,state,assembled=True)
