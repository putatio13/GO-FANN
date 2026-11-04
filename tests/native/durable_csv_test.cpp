#include "profiling/durable_csv.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_all(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

} // namespace

int main()
{
    char path[] = "/tmp/flashanns_durable_csv_test_XXXXXX";
    const int temp_fd = mkstemp(path);
    require(temp_fd >= 0, "mkstemp failed");
    close(temp_fd);

    try {
        const pid_t child = fork();
        require(child >= 0, "fork failed");
        if (child == 0) {
            go_fann::DurableCsvAppender writer(path, "run_index,phase,qps");
            writer.append_row("0,timed,123.5");
            kill(getpid(), SIGKILL); // Simulate interruption before the next run.
            _exit(127);
        }

        int status = 0;
        require(waitpid(child, &status, 0) == child, "waitpid failed");
        require(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL,
                "durable CSV child was not interrupted as expected");
        require(read_all(path) == "run_index,phase,qps\n0,timed,123.5\n",
                "completed sidecar row did not survive immediate process exit");

        std::remove(path);
        std::puts("durable CSV tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::remove(path);
        std::fprintf(stderr, "durable CSV test failed: %s\n", error.what());
        return 1;
    }
}
