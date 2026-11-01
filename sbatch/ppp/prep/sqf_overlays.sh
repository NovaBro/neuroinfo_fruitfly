# Resolve read-only synth_MCFO SquashFS overlays for PPP prep jobs.
# Source from a prep sbatch after DATASETS is set (do not execute directly).
#
# If SQF is set, it is a comma-separated list of *.sqf paths. Otherwise each
# name in DATASETS mounts ${SQF_ROOT}/<name>.sqf when that file exists.
# Images must live under SQF_ROOT and are mounted at /synth_MCFO/<basename>.
#
# Sets:
#   SQF_OVERLAY_ARGS  extra singularity --overlay arguments (may be empty)
#   SQF_MOUNT_CHECK   bash snippet; require /synth_MCFO/<name>/samples
#
# Exits 1 if a volumes/raw symlink under fisbe/<dataset>/{train,val,test}
# points at /synth_MCFO/<name>/ and that image is not being mounted.

SQF_ROOT="/scratch/vh2308/data/synth_MCFO"

sqf_resolve_overlays() {
  local root_real sqf sqf_real name ds split raw target link_name
  local -a sqf_list=()
  local -a dataset_names=()
  local -A seen_sqf=()
  local -A mounted_name=()

  root_real=$(realpath -e "${SQF_ROOT}")
  SQF_OVERLAY_ARGS=()
  SQF_MOUNT_NAMES=()
  SQF_MOUNT_CHECK=""

  IFS=',' read -r -a dataset_names <<< "${DATASETS:-}"

  if [[ -n "${SQF:-}" ]]; then
    IFS=',' read -r -a sqf_list <<< "${SQF}"
  else
    for ds in "${dataset_names[@]}"; do
      ds="${ds// /}"
      [[ -n "${ds}" ]] || continue
      if [[ -f "${SQF_ROOT}/${ds}.sqf" ]]; then
        sqf_list+=("${SQF_ROOT}/${ds}.sqf")
      fi
    done
  fi

  if ((${#sqf_list[@]})); then
    for sqf in "${sqf_list[@]}"; do
      sqf="${sqf// /}"
      [[ -n "${sqf}" ]] || continue
      if [[ "${sqf}" != *.sqf || ! -f "${sqf}" ]]; then
        echo "SQF must be an existing .sqf file: ${sqf}" >&2
        exit 1
      fi
      sqf_real=$(realpath -e "${sqf}")
      case "${sqf_real}" in
        "${root_real}"/*) ;;
        *)
          echo "SQF must be under ${SQF_ROOT}: ${sqf_real}" >&2
          exit 1
          ;;
      esac
      name=$(basename "${sqf_real}" .sqf)
      if ! [[ "${name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "SQF basename is not a usable dataset name: ${name}" >&2
        exit 1
      fi
      if [[ -n "${seen_sqf[${sqf_real}]:-}" ]]; then
        continue
      fi
      seen_sqf["${sqf_real}"]=1
      mounted_name["${name}"]=1
      SQF_OVERLAY_ARGS+=(--overlay "${sqf_real}:ro")
      SQF_MOUNT_NAMES+=("${name}")
      echo "SQF=${sqf_real} mount=/synth_MCFO/${name}"
    done
  fi

  if ((${#SQF_MOUNT_NAMES[@]})); then
    for name in "${SQF_MOUNT_NAMES[@]}"; do
      SQF_MOUNT_CHECK+="if [[ ! -d /synth_MCFO/${name}/samples ]]; then echo \"missing /synth_MCFO/${name}/samples (sqf did not mount at the expected path)\" >&2; exit 1; fi; "
    done
  fi

  for ds in "${dataset_names[@]}"; do
    ds="${ds// /}"
    [[ -n "${ds}" ]] || continue
    for split in train val test; do
      for raw in "fisbe/${ds}/${split}/"*.zarr/volumes/raw; do
        [[ -L "${raw}" ]] || continue
        target=$(readlink "${raw}")
        if [[ "${target}" =~ ^/synth_MCFO/([^/]+)/ ]]; then
          link_name="${BASH_REMATCH[1]}"
          if [[ -z "${mounted_name[${link_name}]:-}" ]]; then
            echo "error: ${raw} -> ${target}" >&2
            echo "not mounted: ${SQF_ROOT}/${link_name}.sqf" >&2
            exit 1
          fi
        fi
      done
    done
  done
}
