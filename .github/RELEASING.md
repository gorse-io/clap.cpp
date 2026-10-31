# Publishing language packages

Publishing a GitHub release runs `.github/workflows/release.yml`. Rust and Node
publishers are independent of the existing Python and CLI release jobs.

Use a SemVer tag such as `v0.2.0` or `v0.2.0-rc.1`. Rust and npm versions are
set from the tag (without `v`); the Node lockfile is updated too. Build metadata
(`+...`) is rejected because npm strips it. npm prereleases use the `next`
dist-tag; stable releases use `latest`. A release marked as a GitHub prerelease
also uses `next`, even if its version has no prerelease suffix.

Both packages ship portable CPU native sources and compile on installation;
this does not publish GPU bindings or prebuilt Node binaries. Before publication,
source archives are built and tested on Linux x64/ARM64, macOS ARM64 and Windows
x64. The publishers download the verified Linux-built archive only after every
platform succeeds. npm publishes that tarball directly. Cargo repackages the
extracted crate before upload: the publisher verifies that all files, including
the normalized manifest and native sources, match the tested archive. Only
Cargo-generated `Cargo.toml.orig` and VCS metadata are excluded from this comparison;
Rust guarantees source equivalence, not a byte-identical compressed archive.

## Registry authorization

Configure GitHub environments `crates-io` and `npm`, including any required
reviewers. The workflow never creates registry credentials or configures trusted
publishers for you.

- **crates.io:** configure a GitHub trusted publisher for repository
  `gorse-io/clap.cpp`, workflow `release.yml`, environment `crates-io`.
  Alternatively, set environment/repository secret `CARGO_REGISTRY_TOKEN` with
  permission to publish `clap-cpp` (including first publication when needed).
  When this secret is set it is used instead of OIDC.
- **npm:** configure a GitHub trusted publisher for repository
  `gorse-io/clap.cpp`, workflow `release.yml`, environment `npm`.
  Alternatively, set environment/repository secret `NPM_TOKEN` to a granular
  publish token authorized for `clap-cpp`, with the permissions/2FA bypass
  required for unattended publication. npm 11 is used for OIDC support;
  publications include provenance. Enable **direct publishing** for this trusted
  publisher because the workflow runs `npm publish`, not staged publication.
  A first publication may require a token
  before the package's trusted publisher can be configured.

Check ownership and name availability before the first release. Neither registry
is contacted with a publish command by ordinary PR CI. Actual publication is
only exercised when a release is published and registry authorization is ready.

## Local checks

```sh
python -m unittest discover -s .github/scripts -v
python rust/scripts/vendor.py
cargo package --manifest-path rust/Cargo.toml --allow-dirty
cd node
npm ci
npm run test:fixtures
npm test
npm run test:types
npm run test:packaging
mkdir -p artifacts
npm pack --pack-destination artifacts
npm install --prefix installed ./artifacts/*.tgz
cd installed/node_modules/clap-cpp
npm run test:fixtures
npm test
```

The checkout must contain the pinned initialized submodules. The Rust vendoring
step copies the required sources and licenses into ignored generated directories;
see `rust/README.md`. Node's `prepack` does the equivalent. Do not commit generated
native trees or archives. CMake, a C++20 compiler and Python are needed for local
checks; install NumPy for Node's numerical oracle.
