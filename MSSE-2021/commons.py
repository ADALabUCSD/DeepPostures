# Copyright 2021 Supun Nakandala. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import h5py
import numpy as np
from datetime import datetime, timedelta

def input_iterator(data_root, subject_id):
    fnames = [name.split('.')[0] for name in os.listdir(os.path.join(data_root, subject_id)) if not name.startswith('.')]
    fnames.sort()
    for i in range(len(fnames) - 1):
        assert datetime.strptime(fnames[i+1], "%Y-%m-%d").date() - datetime.strptime(fnames[i], "%Y-%m-%d").date() == timedelta(days=1)
    
    for fname in fnames:
        h5f = h5py.File(os.path.join(data_root, subject_id,  '{}.h5'.format(fname)), 'r')
        timestamps = h5f.get('time')[:]
        data = h5f.get('data')[:]
        sleeping = h5f.get('sleeping')[:]
        non_wear = h5f.get('non_wear')[:]

        data_batch = []
        timestamps_batch = []
        for d, t, s, nw in zip(data, timestamps, sleeping, non_wear):
            if s == 1 or nw == 1:
                if len(timestamps_batch) > 0:
                    yield np.array(data_batch), np.array(timestamps_batch)
                data_batch = []
                timestamps_batch = []
                continue

            data_batch.append(d)
            timestamps_batch.append(t)

        if len(timestamps_batch) > 0:
            yield np.array(data_batch), np.array(timestamps_batch)

        h5f.close()
