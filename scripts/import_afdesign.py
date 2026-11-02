"""Import genuine discrete ColabDesign trajectories into existing real-data tools."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import Bio
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.SeqUtils import seq1

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from changjiang_select.data import load_candidates,save_candidates,save_csv
from scripts.evaluate_designs import geometry,sequence_metrics
from scripts.import_boltz import ipsae


def main(args):
    run=(ROOT/args.run).resolve()
    meta=json.loads((run/'run.json').read_text())
    if meta['synthetic'] or meta['status']!='generated_not_independently_validated':
        raise ValueError('requires finished actual generation, not debugging or intermediate soft sequences')
    out=(ROOT/args.out).resolve()
    if (out/'candidates.json').exists() and not args.append:
        raise ValueError('choose a fresh dataset or append; preserve independent predictions')
    existing=load_candidates(out/'candidates.json') if args.append else None
    if existing and any(v['version']!=Bio.__version__+' / local-v1'
                        for k,v in existing['metric_definitions'].items() if k.startswith('sequence.')):
        raise ValueError('append using the original sequence-metric BioPython environment')
    for name in ('generation','poses'):
        (out/name).mkdir(parents=True,exist_ok=True)
    baseline=load_candidates(ROOT/'results/tl1a_boltzgen_20261101/candidates.json')
    defs={k:v for k,v in baseline['metric_definitions'].items() if k.startswith(('sequence.','geometry.')) or k=='official.ipsae'}
    for key,meaning,unit,direction in [
        ('binder_plddt','generator binder confidence with native target template','fraction','higher'),
        ('iptm','generator composite-target/binder ipTM; template supplied; actual context in provenance','fraction','higher'),
        ('ipsae_10','generator ipSAE PAE/distance 10 A, composite-target/binder; context in provenance','score','higher'),
        ('ipsae_15','generator ipSAE PAE/distance 15 A, composite-target/binder; context in provenance','score','higher'),
        ('target_beta_CA_rmsd','generator target core RMSD to native before replacing target context','angstrom','lower')]:
        defs['afdesign.'+key]=dict(model='ColabDesign AF2 gradient generation',
            version=meta['upstream_revision']+'; model 3 multimer v3; ipSAE 6174cf9',
            meaning=meaning+'; NOT independent validation or official screening',units=unit,direction=direction)
    if existing:
        defs.update(existing['metric_definitions'])
    for key,spec in defs.items():
        if key.startswith('sequence.'):
            spec['version']=Bio.__version__+' / local-v1'
    meta['import_environment']=dict(biopython=Bio.__version__,numpy=np.__version__)
    parser=PDBParser(QUIET=True)
    native=parser.get_structure('native',ROOT/'targets/TL1A/observed.pdb')
    expected=''.join((ROOT/'targets/TL1A/construct.fasta').read_text().splitlines()[1:])[16:]
    if any(''.join(seq1(r.resname) for r in native[0][c])!=expected for c in 'ABC'):
        raise ValueError('native trimer does not match the official construct at 88-251')
    target_map=meta.get('target_residue_map',[dict(chain='A',uniprot=n) for n in range(88,252)])
    target_len=len(target_map)
    expected=''.join(seq1(native[0][r['chain']][r['uniprot']].resname) for r in target_map)
    manifest=json.loads((ROOT/'targets/manifest.json').read_text())
    core=next(t['beta_core_uniprot'] for t in manifest['targets'] if t['target_id']=='TL1A')
    rows=existing['candidates'] if existing else []
    sequences={r['sequence'] for r in rows}
    imported=[]
    for trial in meta['trajectories']:
        if trial['status']!='generated_not_independently_validated':
            continue
        source=run/f"seed_{trial['seed']}"
        raw=parser.get_structure('generated',source/'generated.pdb')
        if len(list(raw[0]))!=2 or ''.join(seq1(r.resname) for r in raw[0]['A'])!=expected:
            raise ValueError('generator target sequence or chain cardinality mismatch')
        sequence=''.join(seq1(r.resname) for r in raw[0]['B'])
        if sequence!=trial['sequence'] or not 40<=len(sequence)<=100:
            raise ValueError('generated coordinates do not match the final discrete sequence')
        if sequence in sequences:
            raise ValueError('duplicate output; record explicitly rather than create another candidate')
        confidence=json.loads((source/'scores.json').read_text())
        if np.array(confidence['pae']).shape!=(target_len+len(sequence),)*2:
            raise ValueError('generator PAE ordering/length mismatch')
        scores={}; pair_scores={}
        for cutoff in (10,15):
            scores[f'afdesign.ipsae_{cutoff}'],pair_scores[str(cutoff)]=ipsae(
                source/'scores.json',source/'generated.pdb',cutoff,'B','A')
        fit=Superimposer()
        indices=[i for i,r in enumerate(target_map) if r['uniprot'] in core]
        target_residues=list(raw[0]['A'])  # Preserve gaps/chain breaks in original model output.
        fit.set_atoms([native[0][target_map[i]['chain']][target_map[i]['uniprot']]['CA'] for i in indices],
                      [target_residues[i]['CA'] for i in indices])
        fit.apply(list(raw.get_atoms()))
        pose=copy.deepcopy(native)
        binder=raw[0]['B'].copy()
        binder.id='D'
        for r in binder:
            for a in list(r):
                if a.name not in ('N','CA','C','O'):
                    r.detach_child(a.id)
        pose[0].add(binder)
        name=f"tl1a_af_{meta['site']}_{trial['seed']}_gen"
        filename=out/'poses'/f'{name}_design.pdb'
        writer=PDBIO();writer.set_structure(pose);writer.save(str(filename))
        canonical,metrics,epitope=geometry(filename,native,'D')
        writer.set_structure(canonical);writer.save(str(filename))
        bx=np.array([a.coord for a in canonical[0]['D'].get_atoms()])
        nx=np.array([a.coord for a in native.get_atoms() if a.element!='H'])
        metrics['geometry.native_all_chain_BB_clash_pairs2']=int((np.linalg.norm(bx[:,None]-nx,axis=-1)<2).sum())
        metrics.update(sequence_metrics(sequence));metrics.update(scores)
        metrics.update({'official.ipsae':None,'afdesign.binder_plddt':float(np.mean(confidence['plddt'][target_len:]))/100,
            'afdesign.iptm':confidence['iptm'],'afdesign.target_beta_CA_rmsd':float(fit.rms)})
        destination=out/'generation'/f"seed_{trial['seed']}"
        destination.mkdir(parents=True,exist_ok=True)
        for f in ('generated.pdb','scores.json','sequence.fasta','trajectory.json'):
            shutil.copyfile(source/f,destination/f)
        if (run/'design_context.pdb').exists():
            shutil.copyfile(run/'design_context.pdb',out/'generation'/f'{run.name}_context.pdb')
        rows.append(dict(id=name,sequence=sequence,target_id='TL1A',design_type='minibinder',
            source_model='ColabDesign AF2-Multimer gradient hallucination',source_version=meta['upstream_revision'],
            run_id=run.name,structural_group=f"afdesign_{meta['site']}_{trial['seed']}",
            epitope_residues=epitope,metrics=metrics,
            structures=[dict(path=filename.relative_to(ROOT).as_posix(),chain_id='D',role='generated_backbone_complex_not_refolded')],
            metadata=dict(status='generated_not_independently_validated',
                generation=dict(path=(destination/'generated.pdb').relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256((source/'generated.pdb').read_bytes()).hexdigest(),
                    seed=trial['seed'],target_template=True,binder_template=False,
                    source_target_chain='A',source_binder_chain='B',generator_ipsae_pairs=pair_scores),
                coordinates='native full ABC context + generated binder BACKBONE; original all-atom model retained separately')))
        sequences.add(sequence);imported.append(name)
    if not imported:
        raise ValueError('no completed actual trajectory to import')
    dataset=dict(schema_version=1,synthetic=False,metric_definitions=defs,candidates=rows)
    save_candidates(dataset,out/'candidates.json');save_csv(dataset,out/'candidates.csv')
    (out/'candidates.fasta').write_text(''.join(f">{r['id']} generated_unvalidated\n{r['sequence']}\n" for r in rows))
    history=json.loads((out/'generation.json').read_text()) if existing else {'runs':{}}
    history['runs'][run.name]=meta
    (out/'generation.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(imported=imported,distinct_sequences=len(rows)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--out',default='results/tl1a_afdesign_20261102')
    p.add_argument('--append',action='store_true')
    main(p.parse_args())
