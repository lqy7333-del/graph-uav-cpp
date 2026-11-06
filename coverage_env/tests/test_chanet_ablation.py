import argparse
import copy
import json
import pytest
import torch
from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.env.fixed_instance_pool import write_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.encoder.node_mlp import NodeMLPEncoder
from coverage_env.rl.encoder.batched_encoder import collate_graphs, encode_graph_batch
from coverage_env.rl.graph.graph_builder import GraphBuilder
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.chanet_ablation import initialize_from_a_zero

@pytest.fixture
def setup():
    torch.set_num_threads(1)
    c=load_config()
    c['environment']['num_fields']={'min':10,'max':10}
    c['field_generator']['layout']={'mode':'scattered'}
    c['field_generator']['field_size']={'min_area':500.,'max_area':800.}
    c['ppo'].update(epochs=1,batch_size=64,gamma=1.,lr_schedule='none',randomize_uav=False,
                    actor_learning_rate=1e-4,critic_learning_rate=.003)
    c['rl']['reward']['mode']='time'
    c['rl']['decoder']['candidate_attention']=False
    envs=[InstanceGenerator(c).generate(num_fields=10,seed=i+81000,randomize_each_reset=False,randomize_uav=False) for i in range(2)]
    return c,envs


def test_ablation_edges_batch_gradients_and_size(setup):
    c,envs=setup
    graphs=[GraphBuilder(c).build_graph(e) for e in envs]
    a=CoverageGraphEncoder.from_config(c['rl']['encoder'])
    b=CoverageGraphEncoder.from_config({**c['rl']['encoder'],'variant':'node_mlp'})
    a.eval(); b.eval()
    assert isinstance(b,NodeMLPEncoder)
    na,nb=[sum(p.numel() for p in model.parameters()) for model in (a,b)]
    assert abs(nb-na)/na < .01
    g=graphs[0]; changed=copy.deepcopy(g); changed.edge_attr=changed.edge_attr+5
    assert not torch.allclose(a(g)['boundary_embedding'],a(changed)['boundary_embedding'])
    assert torch.equal(b(g)['boundary_embedding'],b(changed)['boundary_embedding'])
    changed.edge_index=changed.edge_index[:,:0]; changed.edge_type=changed.edge_type[:0]; changed.edge_attr=changed.edge_attr[:0]
    assert torch.equal(b(g)['field_embedding'],b(changed)['field_embedding'])
    single=[b(g) for g in graphs]; batch=collate_graphs(graphs); joined=encode_graph_batch(b,batch)
    for i,row in enumerate(single):
        for key,offsets in [('field_embedding',batch.field_offsets),('boundary_embedding',batch.boundary_offsets)]:
            assert torch.allclose(row[key],joined[key][offsets[i]:offsets[i+1]],atol=1e-5)
    loss=sum((joined[k]*torch.randn_like(joined[k])).sum() for k in ('field_embedding','boundary_embedding'))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())


def test_policy_single_batch_and_pairing(setup,tmp_path):
    c,envs=setup
    a=PPOAgent(CoveragePolicy(envs[0],config=c),c,device='cpu')
    payload=a.checkpoint_payload(); payload.update(update_idx=0,training_metadata={'initialization':None,'seed':42})
    path=tmp_path/'a0.pt'; torch.save(payload,path)
    bc=copy.deepcopy(c);bc['rl']['encoder']['variant']='node_mlp'
    b=PPOAgent(CoveragePolicy(envs[0],config=bc),bc,device='cpu')
    meta=initialize_from_a_zero(b,path,bc,{'seed':42})
    assert meta['copied_actor_tensors']>0 and not b.actor_optimizer.state
    for k,v in a.actor.state_dict().items():
        if k in b.actor.state_dict(): assert torch.equal(v,b.actor.state_dict()[k])
    b.actor.eval()
    contexts=[]
    for e in envs:
        state=e.get_state();g=b.actor.graph_builder.build_graph(e);acts=e.get_action_candidates();mask=e.get_action_mask(acts)
        contexts.append(dict(observation=state,graph=g,actions=acts,mask=mask))
    rows=b.actor.forward_batched(contexts)
    for ctx,row in zip(contexts,rows):
        one=b.actor(ctx['observation'],graph=ctx['graph'],candidate_actions=ctx['actions'],feasible_mask=ctx['mask'])
        assert torch.allclose(one['action_distribution'],row['action_distribution'],atol=1e-6)
    payload['update_idx']=20;torch.save(payload,path)
    with pytest.raises(ValueError,match='update 0'): initialize_from_a_zero(b,path,bc,{'seed':42})
    payload['update_idx']=0;torch.save(payload,path)
    with pytest.raises(ValueError,match='metadata'): initialize_from_a_zero(b,path,bc,{'seed':43})
    wrong=copy.deepcopy(bc);wrong['ppo']['gamma']=.99
    with pytest.raises(ValueError,match='configs differ'): initialize_from_a_zero(b,path,wrong,{'seed':42})


def test_worker_b_update_restore_and_variant_rejection(setup,tmp_path):
    from coverage_env.rl.trainer.episode_random_worker import _worker, _RUN_DEFAULTS
    c,envs=setup;bank=tmp_path/'bank';write_bank(bank,c,[envs[0]],[81000])
    def args(out,**kw):
        return argparse.Namespace(**{**_RUN_DEFAULTS,'num_updates':2,'episodes_per_rank':1,
            'device':'cpu','output':str(tmp_path/out),'resume':None,'diagnostic_bank':str(bank),
            'validation_bank':str(bank),'checkpoint_every':1,'eval_every':20,**kw})
    _worker(0,1,args('A',stop_at=1))
    reference=str(tmp_path/'A/ckpt_update_0.pt')
    _worker(0,1,args('B',encoder_variant='node_mlp',paired_init=reference,stop_at=1))
    initial=torch.load(tmp_path/'B/ckpt_update_0.pt',weights_only=False)
    first=torch.load(tmp_path/'B/ckpt_update_1.pt',weights_only=False)
    assert any(not torch.equal(initial['actor'][k],first['actor'][k]) for k in first['actor'])
    assert first['training_metadata']['encoder_variant']=='node_mlp'
    assert first['training_metadata']['paired_initialization']['source_update']==0
    _worker(0,1,args('B',resume=str(tmp_path/'B/ckpt_update_1.pt')))
    assert (tmp_path/'B/ckpt_update_2.pt').exists()
    _worker(0,1,args('Bfull',encoder_variant='node_mlp',paired_init=reference))
    restored=torch.load(tmp_path/'B/ckpt_update_2.pt',weights_only=False)
    continuous=torch.load(tmp_path/'Bfull/ckpt_update_2.pt',weights_only=False)
    for kind in ('actor','critic'):
        assert all(torch.equal(restored[kind][k],continuous[kind][k]) for k in restored[kind])
    assert restored['rank_runtime'][0]['instance_stream']==continuous['rank_runtime'][0]['instance_stream']
    with pytest.raises(ValueError,match='variant changed'):
        _worker(0,1,args('bad',resume=str(tmp_path/'B/ckpt_update_1.pt'),encoder_variant='chanet'))


def test_component_c_d_edge_isolation_single_batch_gradients(setup):
    c, envs = setup
    graphs = [GraphBuilder(c).build_graph(e) for e in envs]
    a = CoverageGraphEncoder.from_config(c['rl']['encoder'])
    c_enc = CoverageGraphEncoder.from_config({**c['rl']['encoder'], 'variant':'no_cross_field_boundary'})
    d_enc = CoverageGraphEncoder.from_config({**c['rl']['encoder'], 'variant':'no_intra_boundary'})
    for m in (a,c_enc,d_enc): m.eval()
    # Parameter counts remain within the replacement-width rule.
    na=sum(p.numel() for p in a.parameters())
    for m in (c_enc,d_enc):
        nm=sum(p.numel() for p in m.parameters())
        assert abs(nm-na)/na < .01
    from coverage_env.rl.graph.relation_builder import EDGE_BB_INTER, EDGE_BB_INTRA
    g=graphs[0]
    inter=copy.deepcopy(g); inter.edge_attr=inter.edge_attr.clone()
    inter.edge_attr[g.edge_type==EDGE_BB_INTER] += 5.0
    intra=copy.deepcopy(g); intra.edge_attr=intra.edge_attr.clone()
    intra.edge_attr[g.edge_type==EDGE_BB_INTRA] += 5.0
    for target in (a,c_enc,d_enc):
        target.eval()
        base=target(g)
        # C ignores cross-field edge attributes; A and D can respond.
        c_out=target(inter)
        d_out=target(intra)
        if target is c_enc:
            assert torch.equal(base['boundary_embedding'],c_out['boundary_embedding'])
            assert not torch.equal(base['boundary_embedding'],d_out['boundary_embedding'])
        elif target is d_enc:
            assert torch.equal(base['boundary_embedding'],d_out['boundary_embedding'])
            assert not torch.equal(base['boundary_embedding'],c_out['boundary_embedding'])
        else:
            assert not torch.equal(base['boundary_embedding'],c_out['boundary_embedding'])
            assert not torch.equal(base['boundary_embedding'],d_out['boundary_embedding'])
    # Local replacement: changing another node cannot affect the target row.
    from coverage_env.rl.encoder.component_ablation import NodeLocalResidual, reference_gat_parameter_count
    local=NodeLocalResidual(16, reference_gat_parameter_count(16,4,4), 4, 0.0).eval()
    h=torch.randn(5,16); h2=h.clone(); h2[0]+=3.0
    ei=torch.tensor([[0,1],[2,3]],dtype=torch.long); ea=torch.randn(2,4)
    out,_=local(h,ei,ea); out2,_=local(h2,ei,ea)
    assert torch.equal(out[1:],out2[1:])
    # Single and disjoint-union batched paths agree for both variants.
    batch=collate_graphs(graphs)
    for enc in (c_enc,d_enc):
        singles=[enc(g) for g in graphs]
        joined=encode_graph_batch(enc,batch)
        for i,row in enumerate(singles):
            assert torch.allclose(row['field_embedding'],joined['field_embedding'][batch.field_offsets[i]:batch.field_offsets[i+1]],atol=1e-5)
            assert torch.allclose(row['boundary_embedding'],joined['boundary_embedding'][batch.boundary_offsets[i]:batch.boundary_offsets[i+1]],atol=1e-5)
        loss=sum((joined[k]*torch.randn_like(joined[k])).sum() for k in ('field_embedding','boundary_embedding'))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_component_c_d_paired_initialization_copies_retained_layers(setup,tmp_path):
    c,envs=setup
    a=PPOAgent(CoveragePolicy(envs[0],config=c),c,device='cpu')
    payload=a.checkpoint_payload(); payload.update(update_idx=0,training_metadata={'initialization':None,'seed':42})
    path=tmp_path/'a0.pt'; torch.save(payload,path)
    for variant,replaced in (('no_cross_field_boundary','encoder.boundary_inter_encoder.'),
                              ('no_intra_boundary','encoder.boundary_intra_encoder.')):
        cc=copy.deepcopy(c); cc['rl']['encoder']['variant']=variant
        agent=PPOAgent(CoveragePolicy(envs[0],config=cc),cc,device='cpu')
        meta=initialize_from_a_zero(agent,path,cc,{'seed':42})
        assert meta['encoder_variant']==variant and meta['replaced_modules']==[replaced]
        assert meta['copied_critic'] and not agent.actor_optimizer.state
        for key,value in payload['actor'].items():
            if key.startswith(replaced): continue
            assert key in agent.actor.state_dict() and torch.equal(value,agent.actor.state_dict()[key]),key
        assert any(k.startswith(replaced) for k in agent.actor.state_dict())
