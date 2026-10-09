# smoke_loader.py
import time
import cv2
from loguru import logger
from exps.default.tde_yolox_s import Exp

# CRITICAL FOR WINDOWS: Disable OpenCV internal thread pools
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def test_workers(worker_count):
    logger.info(f"\n==========================================")
    logger.info(f"Testing with data_num_workers = {worker_count}")
    logger.info(f"==========================================")
    
    exp = Exp()
    exp.data_num_workers = worker_count
    
    # FIX: Pass cache_img=None so YOLOX doesn't throw the dataset assertion
    loader = exp.get_data_loader(
        batch_size=exp.batch_size, 
        is_distributed=False, 
        no_aug=False, 
        cache_img=None
    )
    
    logger.info("Initializing iterator (spawning worker processes)...")
    it = iter(loader)
    
    # Warmup loop: ignore the first 5 batches (spawning / pickling overhead)
    warmup_batches = 5
    benchmark_batches = 25
    
    start_time = None
    for i in range(warmup_batches + benchmark_batches):
        if i == warmup_batches:
            start_time = time.time()
            logger.info("Warmup complete. Measuring throughput...")
            
        t0 = time.time()
        batch = next(it)
        iter_time = time.time() - t0
        
        if i >= warmup_batches:
            logger.info(f"  Batch [{i - warmup_batches + 1}/{benchmark_batches}] fetched in {iter_time:.3f}s")
            
    total_time = time.time() - start_time
    avg_speed = total_time / benchmark_batches
    logger.info(f"RESULT: data_num_workers = {worker_count} -> {avg_speed:.3f}s / batch")
    return avg_speed

if __name__ == "__main__":
    for count in [0, 2, 4]:
        try:
            test_workers(count)
        except Exception as e:
            logger.error(f"Failed at worker count {count}: {e}")
            break