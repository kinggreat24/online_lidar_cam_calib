#pragma once

#include <chrono>

namespace phc {
	namespace utils {
		class StopWatch {
		public:
			StopWatch();
			double GetTimeElapse();
		private:
			std::chrono::steady_clock::time_point start_;
		};
	}
}