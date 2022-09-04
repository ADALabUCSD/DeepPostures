import os
import shutil
import argparse
import glob
import shutil
import math

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partition', type=int, default=8
    )
    parser.add_argument(
        '--root', type=str, default="/mnt/nfs/niddk/ACT_30HZ_CSV"
    )
    parser.add_argument(
        '--ext', type=str, default=".csv.gz"
    )
    parser.add_argument(
        '--run', action="store_true"
    )
    args = parser.parse_args()

    # make partitions
    # for i in range(args.partition):
    #     partition_dir = os.path.join(args.root, str(i))
    #     if not os.path.exists(partition_dir):
    #         os.makedirs(partition_dir)

    all_file_lists = sorted(glob.glob(os.path.join(args.root, "*" + args.ext)))
    # print(os.path.join(args.root, "*" + args.ext), all_file_lists)
    partition_size = math.ceil(len(all_file_lists) // args.partition)
    for i, filepaths in enumerate(chunker(all_file_lists, partition_size)):
        new_root = os.path.join(args.root, str(i))
        if args.run and not os.path.exists(new_root):
            os.makedirs(new_root)
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            new_filepath = os.path.join(new_root, filename)
            if args.run:
                shutil.move(filepath, new_filepath)
            print(new_filepath)

    
