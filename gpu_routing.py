"""Opt-in GPU resource tiers; sizes are total residues across both chains."""
from dataclasses import replace


def tier_errors(tiers):
    errors = []
    if not isinstance(tiers, list):
        return ['cluster.gpu_tiers must be a list']
    previous = 0
    names = set()
    allowed = {'name', 'max_total_residues', 'partition', 'gres', 'constraint'}
    for index, tier in enumerate(tiers):
        label = f'cluster.gpu_tiers[{index}]'
        if not isinstance(tier, dict):
            errors.append(f'{label} must be a mapping')
            continue
        if tier.keys() - allowed:
            errors.append(f'{label}: unknown fields {sorted(tier.keys() - allowed, key=str)}')
        if 'constraint' in tier:
            value = tier['constraint']
            if not isinstance(value, str) or not value or any(c.isspace() or c == '\0' for c in value):
                errors.append(f'{label}.constraint must be a nonempty Slurm feature expression without whitespace')
        for field in ('name', 'partition', 'gres'):
            value = tier.get(field)
            if not isinstance(value, str) or not value.strip() or any(c in value for c in '\0\n\r'):
                errors.append(f'{label}.{field} must be a nonempty string without NUL/newlines')
        name = tier.get('name')
        if isinstance(name, str):
            if name in names:
                errors.append(f'{label}.name must be unique')
            names.add(name)
        partition = tier.get('partition')
        if isinstance(partition, str) and any(not p or any(c.isspace() for c in p)
                                             for p in partition.split(',')):
            errors.append(f'{label}.partition must contain comma-separated partition names')
        maximum = tier.get('max_total_residues')
        if index == len(tiers) - 1:
            if maximum is not None:
                errors.append(f'{label}: final tier must omit max_total_residues (or use null)')
        elif type(maximum) is not int or maximum <= previous:
            errors.append(f'{label}.max_total_residues must be a positive, strictly increasing integer')
        else:
            previous = maximum
    return errors


def tier_index(tiers, pair):
    try:
        length = int(pair['total_length'])
        if length <= 0:
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"GPU tier routing requires a positive total_length in af3_lengths.csv for {pair['system_name']}") from None
    for index, tier in enumerate(tiers):
        if tier.get('max_total_residues') is None or length <= tier['max_total_residues']:
            return index
    raise ValueError('GPU tiers must end with an unbounded tier')


def resource_config(config, tier):
    return replace(config, cluster=replace(config.cluster, partition=tier['partition'], gres=tier['gres'],
                                           constraint=tier.get('constraint', config.cluster.constraint)))
