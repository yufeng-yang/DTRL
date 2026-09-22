from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2
import os
import shutil

input_event_file = "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/DTRL-On_e1_20260516_050501/tb/ppo_e1_1/events.out.tfevents.1778933109.TC-DT-FLO159-02.711029.0"

output_dir = "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/DTRL-On_e1_20260516_050501/tb/ppo_e1_1_trimmed_800k"

max_step = 800_000

# Re-create the output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

loader = EventFileLoader(input_event_file)
writer = EventFileWriter(output_dir)

kept = 0
dropped = 0

for event in loader.Load():
    # Some metadata events in TensorBoard event may have step 0 and need to be retained.
    if event.step <= max_step:
        writer.add_event(event)
        kept += 1
    else:
        dropped += 1

writer.close()

print(f"Done.")
print(f"Input file: {input_event_file}")
print(f"Output dir: {output_dir}")
print(f"Kept events: {kept}")
print(f"Dropped events: {dropped}")