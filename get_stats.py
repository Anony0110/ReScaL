from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
from pathlib import Path
import pandas as pd
import os

pandas2ri.activate()

sk = importr('ScottKnottESD')


def rank_for_file(source_path: str, out_path: str):
    source_path = Path(source_path)
    out_path = Path(out_path)

    if source_path.suffix == '.csv':
        df = pd.read_csv(source_path)
        r_sk = sk.sk_esd(df)
        column_order = list(r_sk[3])
        original_ranks = r_sk[1].astype(int)
        max_rank = original_ranks.max()
        new_ranks = max_rank + 1 - original_ranks

        ranking = pd.DataFrame(
            {
                "technique": [df.columns[i - 1] for i in column_order],
                "rank": new_ranks,
            }
        )

        out_path.mkdir(parents=True, exist_ok=True)

        out_rank_path = out_path / source_path.name
        ranking.to_csv(out_rank_path, index=False)

def calculate_statistics(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        results = {}

        for column in df.columns:
            data = df[column]
            median = data.median()
            mean = data.mean()
            std = data.std()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            results[column] = {
                'median': median,
                'iqr': iqr,
                'mean': mean,
                'std': std

            }

        return results

    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: {csv_file_path}")
    except Exception as e:
        raise Exception(f": {str(e)}")

rqs = ['sota', 'ablation','sensitive']

if __name__ == '__main__':
    # get rank
    for rq in rqs:
        root = rf'result/RQ{rqs.index(rq)}/raw_result'
        out_path = rf'result/RQ{rqs.index(rq)}/rank'
        if rq == 'sota':
            for sys_dir in os.listdir(root):
                sys_dir_path = os.path.join(root, sys_dir)
                if os.path.isdir(sys_dir_path):
                    for filename in os.listdir(sys_dir_path):
                        if filename.endswith('.csv'):
                            print(filename)
                            rank_for_file(os.path.join(sys_dir_path, filename), out_path)
        if rq == 'sensitive' or rq == 'ablation':
            for filename in os.listdir(root):
                if filename.endswith('.csv'):
                    print(filename)
                    rank_for_file(os.path.join(root, filename), out_path)

    # get mean & std
    for rq in rqs:
        dir_path = rf'result/RQ{rqs.index(rq)}/raw_result'
        rank_path = rf'result/RQ{rqs.index(rq)}/rank'
        output_dir = rf'result/RQ{rqs.index(rq)}/stats'

        os.makedirs(output_dir, exist_ok=True)

        if rq == 'sota':
            for sys_dir in os.listdir(dir_path):
                sys_dir_path = os.path.join(dir_path, sys_dir)
                for f in os.listdir(sys_dir_path):
                    if f.endswith('.csv'):
                        result = {}
                        file_path = os.path.join(sys_dir_path, f)

                        stats = calculate_statistics(file_path)
                        rank_file_path = os.path.join(rank_path, f)
                        if not os.path.exists(rank_file_path):
                            print(f"rank files no exist: {rank_file_path}")
                            continue
                        ranks = pd.read_csv(rank_file_path)

                        for method, stat in stats.items():
                            if method not in ranks['technique'].values:
                                print(f"{method}not finded in {rank_file_path}")
                                continue

                            mean = round(stat['mean'], 4)
                            std = round(stat['std'], 4)
                            r = ranks.loc[ranks['technique'] == method, 'rank'].values[0]
                            formatted_str = f'{r}_{mean}_({std})'

                            if method not in result:
                                result[method] = []
                            result[method].append(formatted_str)

                        if result:
                            result_df = pd.DataFrame(result)
                            output_path_temp = os.path.join(output_dir, sys_dir)
                            os.makedirs(output_path_temp, exist_ok=True)
                            output_path = os.path.join(output_path_temp, f'{f}_result.csv')
                            result_df.to_csv(output_path, index=False)
                            print(f"results saved in {output_path}")
                        else:
                            print()
        else:
            for f in os.listdir(dir_path):
                if f.endswith('.csv'):
                    result = {}
                    file_path = os.path.join(dir_path, f)

                    stats = calculate_statistics(file_path)

                    rank_file_path = os.path.join(rank_path, f)
                    if not os.path.exists(rank_file_path):
                        print(f"rank files not found:{rank_file_path}")
                        continue
                    ranks = pd.read_csv(rank_file_path)

                    for method, stat in stats.items():
                        if method not in ranks['technique'].values:
                            print(f"{method} not found in {rank_file_path}")
                            continue

                        mean = round(stat['mean'], 4)
                        std = round(stat['std'], 4)
                        r = ranks.loc[ranks['technique'] == method, 'rank'].values[0]
                        formatted_str = f'{r}_{mean}_({std})'

                        if method not in result:
                            result[method] = []
                        result[method].append(formatted_str)

                    if result:  #
                        result_df = pd.DataFrame(result)
                        output_path = os.path.join(output_dir, f'{f}_result.csv')
                        result_df.to_csv(output_path, index=False)
                        print(f"results saved in: {output_path}")
                    else:
                        print()


