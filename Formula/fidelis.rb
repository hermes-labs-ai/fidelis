# Homebrew formula for fidelis.
#
# Tap install (after the formula is in a tap):
#   brew tap hermes-labs-ai/tap
#   brew install fidelis
#
# Local install (testing):
#   brew install --build-from-source ./Formula/fidelis.rb
#
# This formula installs fidelis as a standalone Python tool via uv (or pip),
# wraps the entry points, and registers a launchd service via `brew services`.

class Fidelis < Formula
  include Language::Python::Virtualenv

  desc "Agent memory with zero-LLM retrieval and a $0-incremental QA scaffold"
  homepage "https://github.com/hermes-labs-ai/fidelis"
  url "https://github.com/hermes-labs-ai/fidelis/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "e8e4df1918d5af4b0625bfaddd9d1e22f21d2bb621ff501ba80ac53c50c64a68"
  license "MIT"
  version "0.1.0"

  depends_on "python@3.12"

  # Resource declarations are managed by `brew update-python-resources Formula/fidelis.rb`
  # once mem0ai and chromadb pin a stable version line.

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"fidelis-server"]
    keep_alive true
    log_path var/"log/fidelis-server.log"
    error_log_path var/"log/fidelis-server.log"
    working_dir HOMEBREW_PREFIX
  end

  test do
    # Confirm the entry points install and respond to --help
    system bin/"fidelis", "--help"
    system bin/"fidelis-server", "--help"
  end
end
