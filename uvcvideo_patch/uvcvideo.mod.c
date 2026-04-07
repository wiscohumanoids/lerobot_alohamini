#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x25f8bfc1, "module_layout" },
	{ 0x3e338b49, "vb2_ioctl_reqbufs" },
	{ 0x2d3385d3, "system_wq" },
	{ 0xc1c7e6c0, "kmalloc_caches" },
	{ 0x7cb2f731, "v4l2_event_unsubscribe" },
	{ 0x3fe2ccbe, "memweight" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xc4f0da12, "ktime_get_with_offset" },
	{ 0x1fdc7df2, "_mcount" },
	{ 0x619cb7dd, "simple_read_from_buffer" },
	{ 0x94c27ac9, "debugfs_create_dir" },
	{ 0x28a41faa, "v4l2_event_queue_fh" },
	{ 0x97d868c2, "_dev_printk" },
	{ 0xf0be113f, "vb2_mmap" },
	{ 0x244e2f1a, "usb_debug_root" },
	{ 0xffcda6b8, "v4l2_device_unregister" },
	{ 0xa9f67a0c, "no_llseek" },
	{ 0xe4efc61a, "vb2_create_bufs" },
	{ 0x12a4e128, "__arch_copy_from_user" },
	{ 0xaf035806, "vb2_fop_poll" },
	{ 0x4ed0370e, "vb2_ioctl_streamon" },
	{ 0x56470118, "__warn_printk" },
	{ 0x3c12dfe, "cancel_work_sync" },
	{ 0xb43f9365, "ktime_get" },
	{ 0xd4bacb8, "usb_kill_urb" },
	{ 0x66b4cc41, "kmemdup" },
	{ 0xc2867216, "vb2_ops_wait_prepare" },
	{ 0xf2d7e632, "__video_register_device" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x1667ecb9, "usb_autopm_get_interface" },
	{ 0x77e3fe5f, "usb_enable_autosuspend" },
	{ 0xe1bf0af5, "debugfs_create_file" },
	{ 0x82737e8, "v4l2_ctrl_merge" },
	{ 0xd7afc0a8, "dma_alloc_noncontiguous" },
	{ 0xdd64e639, "strscpy" },
	{ 0x886b711a, "dma_free_noncontiguous" },
	{ 0x3c3ff9fd, "sprintf" },
	{ 0xc908b39a, "v4l2_device_register" },
	{ 0x65cdc233, "input_event" },
	{ 0xf9c0b663, "strlcat" },
	{ 0xd85c5e47, "_dev_warn" },
	{ 0xdcb764ad, "memset" },
	{ 0x9df33db, "dma_sync_sg_for_device" },
	{ 0xb3103559, "vb2_vmalloc_memops" },
	{ 0x7fdf3a3a, "usb_string" },
	{ 0x4b0a3f52, "gic_nonsecure_priorities" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0xda047559, "vb2_fop_mmap" },
	{ 0x1ccfd5cc, "vb2_ioctl_qbuf" },
	{ 0xd81a27d7, "usb_deregister" },
	{ 0x89940875, "mutex_lock_interruptible" },
	{ 0xbb10749, "v4l2_event_subscribe" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0xb77b0159, "v4l2_prio_init" },
	{ 0xd15bb43e, "video_unregister_device" },
	{ 0xb249fdcf, "media_entity_pads_init" },
	{ 0x19401476, "usb_set_interface" },
	{ 0xbd9ad652, "v4l2_fh_init" },
	{ 0x9bd32252, "vb2_plane_vaddr" },
	{ 0xa2bd2936, "vb2_buffer_done" },
	{ 0xaafdc258, "strcasecmp" },
	{ 0x23d83250, "usb_poison_urb" },
	{ 0x4b750f53, "_raw_spin_unlock_irq" },
	{ 0x27a2116c, "usb_control_msg" },
	{ 0x43cfdd90, "debugfs_remove" },
	{ 0x6a984990, "usb_driver_claim_interface" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x8c03d20c, "destroy_workqueue" },
	{ 0x6277a698, "vb2_qbuf" },
	{ 0x1e55ff75, "vb2_ioctl_prepare_buf" },
	{ 0xc2f701ec, "vb2_ioctl_create_bufs" },
	{ 0x72168581, "dma_vmap_noncontiguous" },
	{ 0xac67a27c, "vb2_querybuf" },
	{ 0x89c9a638, "_dev_err" },
	{ 0x4ca7ab74, "vb2_ioctl_dqbuf" },
	{ 0x42160169, "flush_workqueue" },
	{ 0xf5a53e3f, "media_device_cleanup" },
	{ 0x6fff261f, "__arch_clear_user" },
	{ 0x123959a1, "v4l2_type_names" },
	{ 0x85fd2dbe, "_dev_info" },
	{ 0x46834faf, "usb_submit_urb" },
	{ 0xf50fecbc, "v4l2_ctrl_replace" },
	{ 0x9e59cd30, "vb2_streamon" },
	{ 0x7f2455f4, "usb_get_dev" },
	{ 0xa916b694, "strnlen" },
	{ 0xdc11e457, "vb2_fop_release" },
	{ 0x6cbbfc54, "__arch_copy_to_user" },
	{ 0xba8e127d, "video_devdata" },
	{ 0x296695f, "refcount_warn_saturate" },
	{ 0x3ea1b6e4, "__stack_chk_fail" },
	{ 0xbaa0ee0c, "vb2_expbuf" },
	{ 0x8818600b, "input_register_device" },
	{ 0x9e3c18c9, "usb_put_dev" },
	{ 0xb8b9f817, "kmalloc_order_trace" },
	{ 0x96b29254, "strncasecmp" },
	{ 0x300209fa, "usb_clear_halt" },
	{ 0x8427cc7b, "_raw_spin_lock_irq" },
	{ 0x92997ed8, "_printk" },
	{ 0x908e5601, "cpu_hwcaps" },
	{ 0xc442c16a, "usb_driver_release_interface" },
	{ 0x942b7c9b, "input_free_device" },
	{ 0x32d43420, "v4l2_ctrl_get_name" },
	{ 0x1753fe74, "v4l2_device_register_subdev" },
	{ 0x69f38847, "cpu_hwcap_keys" },
	{ 0x5ee605ff, "media_create_pad_link" },
	{ 0xf2c96e2e, "vb2_reqbufs" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xd58d4b67, "kmem_cache_alloc_trace" },
	{ 0x30f3d94d, "usb_get_intf" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0xa89779a6, "v4l2_fh_open" },
	{ 0xb3a92b36, "devm_gpiod_get_optional" },
	{ 0x4cbb9cab, "v4l2_subdev_init" },
	{ 0xba006f48, "vb2_ioctl_querybuf" },
	{ 0x95596698, "__media_device_register" },
	{ 0x389ad119, "vb2_dqbuf" },
	{ 0x37a0cba, "kfree" },
	{ 0x4829a47e, "memcpy" },
	{ 0xf35ce714, "input_unregister_device" },
	{ 0xae5c7863, "gpiod_to_irq" },
	{ 0xd854665, "usb_match_one_id" },
	{ 0x212708ff, "dma_sync_sg_for_cpu" },
	{ 0x96848186, "scnprintf" },
	{ 0x83f07a09, "usb_register_driver" },
	{ 0x10188a29, "vb2_ops_wait_finish" },
	{ 0x97e80904, "v4l2_fh_add" },
	{ 0x7bd001e3, "dma_vunmap_noncontiguous" },
	{ 0x81afc0cb, "v4l2_fh_del" },
	{ 0x9291cd3b, "memdup_user" },
	{ 0xed1477f7, "usb_ifnum_to_if" },
	{ 0xc5b6f236, "queue_work_on" },
	{ 0x656e4a6e, "snprintf" },
	{ 0x1fdd7232, "vb2_poll" },
	{ 0xdbccf666, "media_device_init" },
	{ 0x90ef03dc, "usb_get_current_frame_number" },
	{ 0x28b12cc9, "v4l2_format_info" },
	{ 0x331591b7, "vb2_ioctl_streamoff" },
	{ 0x5b6b1c4e, "vb2_queue_release" },
	{ 0x283fa325, "param_ops_uint" },
	{ 0x8abf92ad, "devm_request_threaded_irq" },
	{ 0x14b89635, "arm64_const_caps_ready" },
	{ 0x1b98ba5, "vb2_streamoff" },
	{ 0x865f28c0, "usb_free_urb" },
	{ 0x49cd25ed, "alloc_workqueue" },
	{ 0x83d6d404, "media_device_unregister" },
	{ 0x182d34b7, "video_ioctl2" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xd2cdc483, "usb_autopm_put_interface" },
	{ 0x7504e97d, "usb_alloc_urb" },
	{ 0x652f7777, "usb_put_intf" },
	{ 0xe914e41e, "strcpy" },
	{ 0x396dc198, "gpiod_get_value_cansleep" },
	{ 0x377d5bb8, "v4l2_fh_exit" },
	{ 0x363e670, "input_allocate_device" },
	{ 0xf25aa4bf, "vb2_queue_init" },
};

MODULE_INFO(depends, "videobuf2-v4l2,videodev,videobuf2-common,videobuf2-vmalloc,mc");

MODULE_ALIAS("usb:v0416pA91Ad*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0458p706Ed*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v045Ep00F8d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v045Ep0721d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v045Ep0723d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v046Dp0821d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v046Dp0823d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C1d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C2d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C3d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C5d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C6d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp08C7d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v046Dp082Dd*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v04F2pB071d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v058Fp3820d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05A9p2640d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05A9p2641d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05A9p2643d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05A9p264Ad*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05A9p7670d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05ACp8501d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05ACp8600d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05C8p0403d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v05E3p0505d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v06F8p300Cd*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0AC8p332Dd*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0AC8p3410d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0AC8p3420d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0BD3p0555d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v0E8Dp0004d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v13D3p5103d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v152Dp0310d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp5212d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp5931d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp8A12d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp8A31d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp8A33d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v174Fp8A34d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v17DCp0202d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v17EFp480Bd*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1871p0306d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1871p0516d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v18CDpCAFEd*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v18ECp3188d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v18ECp3288d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v18ECp3290d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v199Ep8102d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v19ABp1000d012[0-6]dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v19ABp1000d01[0-1]*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v19ABp1000d00*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1B3Bp2951d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1B3Fp2002d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1BCFp0B40d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v1C4Fp3000d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v2833p0201d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v2833p0211d*dc*dsc*dp*icFFisc01ip00in*");
MODULE_ALIAS("usb:v29FEp4D53d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v8086p0B03d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v*p*d*dc*dsc*dp*ic0Eisc01ip00in*");
MODULE_ALIAS("usb:v*p*d*dc*dsc*dp*ic0Eisc01ip01in*");

MODULE_INFO(srcversion, "C2F7D8AB1E486A98D70A3E4");
