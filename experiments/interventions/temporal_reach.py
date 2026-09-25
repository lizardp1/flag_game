"""Schedule-only temporal reach. No model calls or causal-accuracy claims."""
import argparse
import json
from pathlib import Path


def first_arrivals(events, source, n):
    reached={source:0}
    for event in sorted(events,key=lambda e:e['t']):
        speaker,listener=int(event['speaker_id']),int(event['listener_id'])
        if not 0 <= speaker < n or not 0 <= listener < n or event['t'] <= 0:
            raise ValueError('Invalid contact IDs or time')
        if speaker in reached and listener not in reached:
            reached[listener]=event['t']
    return {str(i):reached.get(i) for i in range(n)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interactions',type=Path,required=True)
    p.add_argument('--N',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--valid-only',action='store_true')
    a=p.parse_args()
    if a.N<2:raise ValueError('N must be >=2')
    events=[json.loads(line) for line in a.interactions.read_text().splitlines() if line.strip()]
    if a.valid_only:events=[e for e in events if e.get('valid')]
    times=[e['t'] for e in events]
    if len(set(times))!=len(times):raise ValueError('Expected unique asynchronous interaction times')
    results={}
    for source in range(a.N):
        arrivals=first_arrivals(events,source,a.N)
        score=sum(a.N/t for agent,t in arrivals.items() if int(agent)!=source and t is not None)/(a.N-1)
        results[str(source)]={'first_arrival_interactions':arrivals,'mean_inverse_arrival_rounds':score}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'interpretation':'Schedule-only transmission opportunities; not causal attribution','valid_only':a.valid_only,'agents':results},indent=2)+'\n')

if __name__=='__main__':main()
