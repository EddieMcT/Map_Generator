import cProfile
import memory_profiler
import time
import numpy as np
from noise_functions import my_perl
from landscape import landscape_gen

class PerlinProfiler:
    def __init__(self):
        self.x = np.linspace(0, 10, 1024)
        self.y = np.linspace(0, 10, 1024)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    @memory_profiler.profile
    def profile_single_sample(self, voron = False):
        return my_perl.sample(self.X, self.Y, neg_octaves=4, octaves=4, ndims=2, voron=voron)
    
    def time_complexity_test(self):
        sizes = [128, 256, 512, 1024, 1536, 2048]#, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
        times = []
        memories = []
        for size in sizes:
            x = np.linspace(0, 10, size)
            y = np.linspace(0, 10, size)
            X, Y = np.meshgrid(x, y)
            
            start_time = time.perf_counter()
            _ = my_perl.sample(X, Y, neg_octaves=2, octaves=1, ndims=2)
            times.append(time.perf_counter() - start_time)
            
            # Track peak memory for this size
            mem_usage = memory_profiler.memory_usage((my_perl.sample, (X, Y), 
                {'neg_octaves': 2, 'octaves': 1, 'ndims': 3}), max_usage=True)
            memories.append(mem_usage)
            
        return sizes, times, memories
    
class LandscapeProfiler:
    def __init__(self):
        self.landscape = landscape_gen()
        self.x = np.linspace(0, 10, 1024)
        self.y = np.linspace(0, 10, 1024)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def profile_landscape_generation(self):
        """Profile the full landscape generation process"""
        return self.landscape.get_base_height(self.X, self.Y)
        
    def profile_noise_calls(self):
        # Monkey patch the sample method to count calls
        original_sample = my_perl.sample
        call_count = 0
        
        def counting_sample(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_sample(*args, **kwargs)
        
        my_perl.sample = counting_sample
        
        # Run landscape generation
        x = np.linspace(0, 10, 1024)
        y = np.linspace(0, 10, 1024)
        X, Y = np.meshgrid(x, y)
        _ = self.landscape.get_base_height(X, Y)
        
        # Restore original method
        my_perl.sample = original_sample
        return call_count

    @memory_profiler.profile
    def profile_base_pattern_memory(self):
        # Profile memory usage of pattern_ref
        return my_perl.pattern_ref.nbytes



def run_profiling(full = False):
    # Profile individual Perlin noise sample
    profiler = PerlinProfiler()
    print("\nProfiling single Perlin noise sample:")
    cProfile.runctx('profiler.profile_single_sample()', globals(), {'profiler': profiler})
    print("\nProfiling single Perlin noise sample with voronoi:")
    cProfile.runctx('profiler.profile_single_sample(voron=True)', globals(), {'profiler': profiler})
    
    # Add landscape profiling
    landscape_prof = LandscapeProfiler()
    print("\nProfiling full landscape generation:")
    cProfile.runctx('landscape_prof.profile_landscape_generation()', 
                   globals(), 
                   {'landscape_prof': landscape_prof})
    
    if full:
        # Analyze scaling with size
        sizes, times, memories = profiler.time_complexity_test()
        
        # Plot results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot([i**2 for i in sizes], times)
        plt.title('Time vs Size')
        plt.xlabel('Grid Size (px)')
        plt.ylabel('Time (s)')
        
        plt.subplot(122)
        plt.plot([i**2 for i in sizes], memories)
        plt.title('Memory vs Size')
        plt.xlabel('Grid Size (px)')
        plt.ylabel('Peak Memory (MB)')
        plt.show()
        
        # Profile full landscape generation
        landscape_prof = LandscapeProfiler()
        noise_calls = landscape_prof.profile_noise_calls()
        print(f"Total noise function calls: {noise_calls}")

if __name__ == "__main__":
    run_profiling(False)
